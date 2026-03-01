use super::RagChunkArgs;
use crate::config::AppConfig;
use anyhow::{Context, Result, anyhow};
use chrono::{SecondsFormat, Utc};
use indicatif::{ProgressBar, ProgressStyle};
use serde::Serialize;
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

const MD_ROOT_DIRNAME: &str = "md_data";
const CHUNKS_ROOT_DIRNAME: &str = "chunks";
const BASE_DOCS_URL: &str = "https://developer.auroraos.ru";

const TARGET_CHARS: usize = 1200;
const OVERLAP_CHARS: usize = 120;
const HARD_MAX_CHARS: usize = 1600;
const MIN_CHARS: usize = 80;
const SHARD_MAX_RECORDS: usize = 5000;

macro_rules! vprintln {
    ($verbose:expr, $($arg:tt)*) => {
        if $verbose {
            println!($($arg)*);
        }
    };
}

#[derive(Debug, Clone, Serialize)]
struct ChunkingConfig {
    target_chars: usize,
    overlap_chars: usize,
    hard_max_chars: usize,
    min_chars: usize,
    shard_max_records: usize,
}

#[derive(Debug, Serialize)]
struct ChunkManifest {
    started_at: String,
    finished_at: String,
    input_root: String,
    output_root: String,
    docs_total: usize,
    docs_processed: usize,
    chunks_total: usize,
    shards_total: usize,
    errors_total: usize,
    chunking_config: ChunkingConfig,
    per_bucket: BTreeMap<String, BucketStats>,
    errors: Vec<ChunkErrorRecord>,
}

#[derive(Debug, Default, Serialize)]
struct BucketStats {
    docs_total: usize,
    docs_processed: usize,
    chunks_total: usize,
    errors_total: usize,
}

#[derive(Debug, Serialize)]
struct ChunkErrorRecord {
    source_doc_path: String,
    error: String,
}

#[derive(Debug, Serialize)]
struct ChunkRecord {
    id: String,
    version_bucket: String,
    source_md: String,
    doc_key: String,
    chunk_index: usize,
    chunk_count_in_doc: usize,
    heading_path: Vec<String>,
    section_anchor: Option<String>,
    source_title: String,
    source_url: String,
    source_url_with_anchor: String,
    char_count: usize,
    has_code: bool,
    has_table: bool,
    has_list: bool,
    table_id: Option<String>,
    table_ordinal: Option<usize>,
    row_ordinal: Option<usize>,
    row_key: Option<String>,
    column_headers: Option<Vec<String>>,
    cell_ordinal: Option<usize>,
    cell_part_index: Option<usize>,
    cell_part_total: Option<usize>,
    is_table_continuation: bool,
    content_hash: String,
    content: String,
}

#[derive(Debug, Clone)]
struct InputDoc {
    bucket: String,
    path: PathBuf,
    file_name: String,
    source_doc_path: String,
}

#[derive(Debug, Clone)]
struct Section {
    heading_path: Vec<String>,
    section_anchor: Option<String>,
    source_title: String,
    text: String,
}

#[derive(Debug, Clone, Copy)]
enum BlockKind {
    Text,
    Code,
    Table,
    List,
}

#[derive(Debug, Clone)]
struct Block {
    kind: BlockKind,
    text: String,
    table_info: Option<TableBlockInfo>,
}

#[derive(Debug, Clone)]
struct ChunkDraft {
    content: String,
    heading_path: Vec<String>,
    section_anchor: Option<String>,
    source_title: String,
    has_code: bool,
    has_table: bool,
    has_list: bool,
    table_part: Option<TablePartMeta>,
}

#[derive(Debug, Clone)]
struct TableBlockInfo {
    table_id: String,
    table_ordinal: usize,
    column_headers: Vec<String>,
}

#[derive(Debug, Clone)]
struct TablePartMeta {
    table_id: String,
    table_ordinal: usize,
    row_ordinal: usize,
    row_key: String,
    column_headers: Vec<String>,
    cell_ordinal: usize,
    cell_part_index: usize,
    cell_part_total: usize,
    is_table_continuation: bool,
}

#[derive(Debug, Clone)]
struct BlockPiece {
    text: String,
    table_part: Option<TablePartMeta>,
}

#[derive(Debug, Clone, Copy)]
enum SplitMode {
    Text,
    Line,
}

pub fn run(args: RagChunkArgs) -> Result<()> {
    let verbose = args.verbose;
    let config = ChunkingConfig {
        target_chars: TARGET_CHARS,
        overlap_chars: OVERLAP_CHARS,
        hard_max_chars: HARD_MAX_CHARS,
        min_chars: MIN_CHARS,
        shard_max_records: SHARD_MAX_RECORDS,
    };

    let home_dir = AppConfig::load()?.data_root()?;
    let input_root = home_dir.join(MD_ROOT_DIRNAME);
    let output_root = home_dir.join(CHUNKS_ROOT_DIRNAME);

    if !input_root.exists() {
        return Err(anyhow!(
            "input directory does not exist: {}",
            input_root.display()
        ));
    }
    fs::create_dir_all(&output_root).with_context(|| {
        format!(
            "failed to create output directory: {}",
            output_root.display()
        )
    })?;
    cleanup_output_dir(&output_root)?;

    let files = discover_input_docs(&input_root)?;
    let docs_total = files.len();
    let progress = if verbose {
        let pb = ProgressBar::new(docs_total as u64);
        let style =
            ProgressStyle::with_template("[chunk] [{bar:28.green/white}] {pos}/{len}: {msg}")
                .expect("progress template must be valid")
                .progress_chars("=> ");
        pb.set_style(style);
        Some(pb)
    } else {
        None
    };

    vprintln!(verbose, "[chunk] input: {}", input_root.display());
    vprintln!(verbose, "[chunk] output: {}", output_root.display());
    vprintln!(verbose, "[chunk] docs: {}", docs_total);

    let started_at = now_rfc3339();
    let mut docs_processed = 0usize;
    let mut chunks_total = 0usize;
    let mut shards_total = 0usize;
    let mut errors_total = 0usize;
    let mut errors = Vec::new();
    let mut per_bucket: BTreeMap<String, BucketStats> = BTreeMap::new();

    let mut shard_index = 0usize;
    let mut records_in_current_shard = 0usize;
    let mut writer: Option<BufWriter<File>> = None;

    for doc in &files {
        let bucket_stats = per_bucket.entry(doc.bucket.clone()).or_default();
        bucket_stats.docs_total += 1;

        if let Some(pb) = &progress {
            pb.set_message(shorten_for_progress(&doc.source_doc_path, 100));
        }

        match process_document(doc) {
            Ok(chunks) => {
                let chunk_count_in_doc = chunks.len();
                let source_url = build_source_url(&doc.bucket, &doc.file_name)?;
                let doc_key = doc.file_name.clone();

                for (chunk_index, chunk) in chunks.into_iter().enumerate() {
                    if writer.is_none() || records_in_current_shard >= SHARD_MAX_RECORDS {
                        shard_index += 1;
                        writer = Some(open_shard_writer(&output_root, shard_index)?);
                        records_in_current_shard = 0;
                        shards_total += 1;
                    }

                    let content_hash = sha1_hex(&chunk.content);
                    let source_url_with_anchor =
                        make_source_url_with_anchor(&source_url, chunk.section_anchor.as_deref());
                    let id = sha1_hex(&format!(
                        "{}|{}|{}|{}",
                        doc.bucket, doc.source_doc_path, chunk_index, content_hash
                    ));
                    let char_count = chunk.content.chars().count();

                    let record = ChunkRecord {
                        id,
                        version_bucket: doc.bucket.clone(),
                        source_md: doc.source_doc_path.clone(),
                        doc_key: doc_key.clone(),
                        chunk_index,
                        chunk_count_in_doc,
                        heading_path: chunk.heading_path,
                        section_anchor: chunk.section_anchor,
                        source_title: chunk.source_title,
                        source_url: source_url.clone(),
                        source_url_with_anchor,
                        char_count,
                        has_code: chunk.has_code,
                        has_table: chunk.has_table,
                        has_list: chunk.has_list,
                        table_id: chunk.table_part.as_ref().map(|m| m.table_id.clone()),
                        table_ordinal: chunk.table_part.as_ref().map(|m| m.table_ordinal),
                        row_ordinal: chunk.table_part.as_ref().map(|m| m.row_ordinal),
                        row_key: chunk.table_part.as_ref().map(|m| m.row_key.clone()),
                        column_headers: chunk.table_part.as_ref().map(|m| m.column_headers.clone()),
                        cell_ordinal: chunk.table_part.as_ref().map(|m| m.cell_ordinal),
                        cell_part_index: chunk.table_part.as_ref().map(|m| m.cell_part_index),
                        cell_part_total: chunk.table_part.as_ref().map(|m| m.cell_part_total),
                        is_table_continuation: chunk
                            .table_part
                            .as_ref()
                            .map(|m| m.is_table_continuation)
                            .unwrap_or(false),
                        content_hash,
                        content: chunk.content,
                    };

                    write_jsonl_record(
                        writer
                            .as_mut()
                            .expect("shard writer must be initialized before write"),
                        &record,
                    )?;
                    records_in_current_shard += 1;
                    chunks_total += 1;
                    bucket_stats.chunks_total += 1;
                }

                docs_processed += 1;
                bucket_stats.docs_processed += 1;
            }
            Err(err) => {
                errors_total += 1;
                bucket_stats.errors_total += 1;
                errors.push(ChunkErrorRecord {
                    source_doc_path: doc.source_doc_path.clone(),
                    error: err.to_string(),
                });
                if let Some(pb) = &progress {
                    let doc_path = doc.source_doc_path.clone();
                    let err_string = err.to_string();
                    pb.suspend(|| eprintln!("[chunk][warn] {doc_path}: {err_string}"));
                }
            }
        }

        if let Some(pb) = &progress {
            pb.inc(1);
        }
    }

    if let Some(mut w) = writer {
        w.flush().context("failed to flush chunk shard writer")?;
    }
    if let Some(pb) = progress {
        pb.finish_and_clear();
    }

    let finished_at = now_rfc3339();
    let manifest = ChunkManifest {
        started_at,
        finished_at,
        input_root: input_root.display().to_string(),
        output_root: output_root.display().to_string(),
        docs_total,
        docs_processed,
        chunks_total,
        shards_total,
        errors_total,
        chunking_config: config,
        per_bucket,
        errors,
    };

    let manifest_path = output_root.join("manifest.json");
    let manifest_json =
        serde_json::to_vec_pretty(&manifest).context("failed to serialize chunk manifest")?;
    fs::write(&manifest_path, manifest_json)
        .with_context(|| format!("failed to write manifest: {}", manifest_path.display()))?;

    vprintln!(
        verbose,
        "[chunk] finished: docs_processed={}, chunks_total={}, shards_total={}, errors_total={}, manifest={}",
        docs_processed,
        chunks_total,
        shards_total,
        errors_total,
        manifest_path.display()
    );
    println!(
        "{} chunks created from {} documents and stored in {}",
        chunks_total,
        docs_processed,
        output_root.display()
    );

    Ok(())
}

fn discover_input_docs(input_root: &Path) -> Result<Vec<InputDoc>> {
    let mut docs = Vec::new();
    for entry in fs::read_dir(input_root)
        .with_context(|| format!("failed to read input root: {}", input_root.display()))?
    {
        let entry = entry.with_context(|| {
            format!("failed to read directory entry in {}", input_root.display())
        })?;
        let path = entry.path();
        let ftype = entry.file_type().with_context(|| {
            format!(
                "failed to read file type for directory entry: {}",
                path.display()
            )
        })?;
        if !ftype.is_dir() {
            continue;
        }

        let bucket = entry.file_name().to_string_lossy().to_string();
        if bucket != "default" && !is_version_segment(&bucket) {
            continue;
        }

        for md_entry in fs::read_dir(&path)
            .with_context(|| format!("failed to read bucket directory: {}", path.display()))?
        {
            let md_entry = md_entry.with_context(|| {
                format!(
                    "failed to read directory entry in bucket {}",
                    path.display()
                )
            })?;
            let md_path = md_entry.path();
            let md_type = md_entry.file_type().with_context(|| {
                format!(
                    "failed to read file type for directory entry: {}",
                    md_path.display()
                )
            })?;
            if !md_type.is_file() {
                continue;
            }
            let file_name = md_entry.file_name().to_string_lossy().to_string();
            if !file_name.ends_with(".md") {
                continue;
            }

            docs.push(InputDoc {
                bucket: bucket.clone(),
                path: md_path,
                source_doc_path: format!("{bucket}/{file_name}"),
                file_name,
            });
        }
    }

    docs.sort_by(|a, b| a.source_doc_path.cmp(&b.source_doc_path));
    Ok(docs)
}

fn cleanup_output_dir(output_root: &Path) -> Result<()> {
    for entry in fs::read_dir(output_root)
        .with_context(|| format!("failed to read output directory: {}", output_root.display()))?
    {
        let entry = entry.with_context(|| {
            format!(
                "failed to read directory entry in output directory: {}",
                output_root.display()
            )
        })?;
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        if name == "manifest.json" || (name.starts_with("chunks-") && name.ends_with(".jsonl")) {
            fs::remove_file(&path).with_context(|| {
                format!("failed to remove old chunk artifact: {}", path.display())
            })?;
        }
    }
    Ok(())
}

fn open_shard_writer(output_root: &Path, shard_index: usize) -> Result<BufWriter<File>> {
    let path = output_root.join(format!("chunks-{shard_index:05}.jsonl"));
    let file = File::create(&path)
        .with_context(|| format!("failed to create shard: {}", path.display()))?;
    Ok(BufWriter::new(file))
}

fn write_jsonl_record(writer: &mut BufWriter<File>, record: &ChunkRecord) -> Result<()> {
    serde_json::to_writer(&mut *writer, record).context("failed to encode chunk json record")?;
    writer
        .write_all(b"\n")
        .context("failed to write jsonl newline")?;
    Ok(())
}

fn process_document(doc: &InputDoc) -> Result<Vec<ChunkDraft>> {
    let text = fs::read_to_string(&doc.path)
        .with_context(|| format!("failed to read markdown file: {}", doc.path.display()))?;
    let chunks = chunk_document(&text, &doc.source_doc_path);
    Ok(chunks)
}

fn chunk_document(text: &str, source_doc_path: &str) -> Vec<ChunkDraft> {
    let sections = parse_sections(text);
    let doc_title = sections
        .iter()
        .find_map(|s| s.heading_path.last().cloned())
        .unwrap_or_else(|| "Document".to_string());

    let mut chunks = Vec::new();
    for section in sections {
        let table_seed = format!(
            "{}|{}|{}",
            source_doc_path,
            section.heading_path.join(" > "),
            section.section_anchor.clone().unwrap_or_default()
        );
        let blocks = split_section_into_blocks(&section.text, &table_seed);
        let section_title = if section.source_title.is_empty() {
            doc_title.clone()
        } else {
            section.source_title.clone()
        };
        chunks.extend(pack_section_blocks(
            &blocks,
            &section.heading_path,
            section.section_anchor.clone(),
            &section_title,
        ));
    }
    merge_short_chunks(chunks)
}

fn parse_sections(text: &str) -> Vec<Section> {
    let mut sections = Vec::new();
    let mut current_lines: Vec<String> = Vec::new();
    let mut current_heading_path: Vec<String> = Vec::new();
    let mut current_anchor: Option<String> = None;
    let mut current_title = String::new();

    let mut heading_stack: Vec<(usize, String)> = Vec::new();
    let mut saw_heading = false;

    for raw_line in text.lines() {
        if let Some((level, title, anchor)) = parse_heading_line(raw_line) {
            if !current_lines.is_empty() {
                sections.push(Section {
                    heading_path: current_heading_path.clone(),
                    section_anchor: current_anchor.clone(),
                    source_title: current_title.clone(),
                    text: current_lines.join("\n"),
                });
                current_lines.clear();
            }

            while heading_stack.len() >= level {
                heading_stack.pop();
            }
            heading_stack.push((level, title.clone()));
            current_heading_path = heading_stack.iter().map(|(_, t)| t.clone()).collect();
            current_anchor = anchor;
            current_title = title;
            current_lines.push(raw_line.to_string());
            saw_heading = true;
        } else {
            if !saw_heading && current_heading_path.is_empty() {
                current_title.clear();
            }
            current_lines.push(raw_line.to_string());
        }
    }

    if !current_lines.is_empty() {
        sections.push(Section {
            heading_path: current_heading_path,
            section_anchor: current_anchor,
            source_title: current_title,
            text: current_lines.join("\n"),
        });
    }

    sections
        .into_iter()
        .filter(|s| !s.text.trim().is_empty())
        .collect()
}

fn parse_heading_line(line: &str) -> Option<(usize, String, Option<String>)> {
    let trimmed = line.trim_end();
    let mut level = 0usize;
    for ch in trimmed.chars() {
        if ch == '#' {
            level += 1;
        } else {
            break;
        }
    }
    if !(1..=6).contains(&level) {
        return None;
    }

    let rest = trimmed.get(level..)?;
    if !rest.starts_with(' ') {
        return None;
    }
    let mut heading = rest.trim();
    heading = heading.trim_end_matches('#').trim();
    if heading.is_empty() {
        return None;
    }

    let mut anchor = None;
    let mut title = heading.to_string();
    if title.ends_with('}')
        && title.contains("{#")
        && let Some(idx) = title.rfind("{#")
    {
        let raw_anchor = &title[idx + 2..title.len() - 1];
        if !raw_anchor.trim().is_empty() {
            anchor = Some(raw_anchor.trim().to_string());
            title = title[..idx].trim().to_string();
        }
    }
    if title.is_empty() {
        return None;
    }

    Some((level, title, anchor))
}

fn split_section_into_blocks(section_text: &str, table_seed: &str) -> Vec<Block> {
    let lines: Vec<&str> = section_text.lines().collect();
    let mut blocks = Vec::new();
    let mut idx = 0usize;
    let mut table_ordinal = 0usize;

    while idx < lines.len() {
        let line = lines[idx];
        if line.trim().is_empty() {
            idx += 1;
            continue;
        }

        if line.trim_start().starts_with("```") {
            let mut block_lines = vec![line.to_string()];
            idx += 1;
            while idx < lines.len() {
                block_lines.push(lines[idx].to_string());
                if lines[idx].trim_start().starts_with("```") {
                    idx += 1;
                    break;
                }
                idx += 1;
            }
            blocks.push(Block {
                kind: BlockKind::Code,
                text: block_lines.join("\n"),
                table_info: None,
            });
            continue;
        }

        if is_table_line(line) {
            let mut block_lines = vec![line.to_string()];
            idx += 1;
            while idx < lines.len() && is_table_line(lines[idx]) {
                block_lines.push(lines[idx].to_string());
                idx += 1;
            }
            let header_cells = block_lines
                .first()
                .map(|row| parse_table_cells(row))
                .unwrap_or_default();
            let table_id = sha1_hex(&format!("{table_seed}|table#{table_ordinal}"));
            blocks.push(Block {
                kind: BlockKind::Table,
                text: block_lines.join("\n"),
                table_info: Some(TableBlockInfo {
                    table_id,
                    table_ordinal,
                    column_headers: header_cells,
                }),
            });
            table_ordinal += 1;
            continue;
        }

        if is_list_start(line) {
            let mut block_lines = vec![line.to_string()];
            idx += 1;
            while idx < lines.len() {
                let next = lines[idx];
                if next.trim().is_empty() {
                    break;
                }
                if is_list_start(next) || is_list_continuation(next) {
                    block_lines.push(next.to_string());
                    idx += 1;
                    continue;
                }
                break;
            }
            blocks.push(Block {
                kind: BlockKind::List,
                text: block_lines.join("\n"),
                table_info: None,
            });
            continue;
        }

        let mut block_lines = vec![line.to_string()];
        idx += 1;
        while idx < lines.len() {
            let next = lines[idx];
            if next.trim().is_empty()
                || next.trim_start().starts_with("```")
                || is_table_line(next)
                || is_list_start(next)
            {
                break;
            }
            block_lines.push(next.to_string());
            idx += 1;
        }
        blocks.push(Block {
            kind: BlockKind::Text,
            text: block_lines.join("\n"),
            table_info: None,
        });
    }

    blocks
}

fn is_table_line(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.starts_with('|') && trimmed.ends_with('|')
}

fn is_table_separator_line(line: &str) -> bool {
    let trimmed = line.trim();
    if !trimmed.starts_with('|') || !trimmed.ends_with('|') {
        return false;
    }
    trimmed
        .chars()
        .all(|ch| matches!(ch, '|' | '-' | ':' | ' ' | '\t'))
}

fn is_list_start(line: &str) -> bool {
    let trimmed = line.trim_start();
    if trimmed.starts_with("* ") || trimmed.starts_with("- ") || trimmed.starts_with("+ ") {
        return true;
    }
    is_ordered_list_marker(trimmed)
}

fn is_ordered_list_marker(line: &str) -> bool {
    let mut seen_digit = false;
    for ch in line.chars() {
        if ch.is_ascii_digit() {
            seen_digit = true;
            continue;
        }
        if ch == '.' && seen_digit {
            let rest = &line[line.find('.').unwrap_or(0) + 1..];
            return rest.starts_with(' ');
        }
        return false;
    }
    false
}

fn is_list_continuation(line: &str) -> bool {
    line.starts_with("  ") || line.starts_with('\t')
}

fn pack_section_blocks(
    blocks: &[Block],
    heading_path: &[String],
    section_anchor: Option<String>,
    source_title: &str,
) -> Vec<ChunkDraft> {
    let mut result = Vec::new();
    let mut current: Option<ChunkDraft> = None;

    for block in blocks {
        let block_text = block.text.trim();
        if block_text.is_empty() {
            continue;
        }
        let block_len = block_text.chars().count();
        let (has_code, has_table, has_list) = flags_from_kind(block.kind);

        if block_len <= TARGET_CHARS {
            if let Some(ref mut cur) = current {
                if cur.content.chars().count() + 2 + block_len <= TARGET_CHARS {
                    cur.content.push_str("\n\n");
                    cur.content.push_str(block_text);
                    cur.has_code |= has_code;
                    cur.has_table |= has_table;
                    cur.has_list |= has_list;
                } else {
                    result.push(cur.clone());
                    *cur = build_chunk_draft(
                        block_text.to_string(),
                        heading_path,
                        section_anchor.clone(),
                        source_title.to_string(),
                        has_code,
                        has_table,
                        has_list,
                        None,
                    );
                }
            } else {
                current = Some(build_chunk_draft(
                    block_text.to_string(),
                    heading_path,
                    section_anchor.clone(),
                    source_title.to_string(),
                    has_code,
                    has_table,
                    has_list,
                    None,
                ));
            }
            continue;
        }

        if block_len <= HARD_MAX_CHARS {
            if let Some(cur) = current.take() {
                result.push(cur);
            }
            result.push(build_chunk_draft(
                block_text.to_string(),
                heading_path,
                section_anchor.clone(),
                source_title.to_string(),
                has_code,
                has_table,
                has_list,
                None,
            ));
            continue;
        }

        if let Some(cur) = current.take() {
            result.push(cur);
        }

        for piece in split_oversized_block(block) {
            let piece_trimmed = piece.text.trim();
            if piece_trimmed.is_empty() {
                continue;
            }
            result.push(build_chunk_draft(
                piece_trimmed.to_string(),
                heading_path,
                section_anchor.clone(),
                source_title.to_string(),
                has_code,
                has_table,
                has_list,
                piece.table_part,
            ));
        }
    }

    if let Some(cur) = current {
        result.push(cur);
    }

    result
}

fn build_chunk_draft(
    content: String,
    heading_path: &[String],
    section_anchor: Option<String>,
    source_title: String,
    has_code: bool,
    has_table: bool,
    has_list: bool,
    table_part: Option<TablePartMeta>,
) -> ChunkDraft {
    ChunkDraft {
        content,
        heading_path: heading_path.to_vec(),
        section_anchor,
        source_title,
        has_code,
        has_table,
        has_list,
        table_part,
    }
}

fn flags_from_kind(kind: BlockKind) -> (bool, bool, bool) {
    match kind {
        BlockKind::Code => (true, false, false),
        BlockKind::Table => (false, true, false),
        BlockKind::List => (false, false, true),
        BlockKind::Text => (false, false, false),
    }
}

fn split_oversized_block(block: &Block) -> Vec<BlockPiece> {
    match block.kind {
        BlockKind::Code => split_code_block(&block.text)
            .into_iter()
            .map(|text| BlockPiece {
                text,
                table_part: None,
            })
            .collect(),
        BlockKind::Table => split_table_block(block),
        BlockKind::List => split_list_block(&block.text)
            .into_iter()
            .map(|text| BlockPiece {
                text,
                table_part: None,
            })
            .collect(),
        BlockKind::Text => split_with_preferences(&block.text, SplitMode::Text)
            .into_iter()
            .map(|text| BlockPiece {
                text,
                table_part: None,
            })
            .collect(),
    }
}

fn split_list_block(block: &str) -> Vec<String> {
    let items = extract_list_items(block);
    if items.len() < 2 {
        return split_with_preferences(block, SplitMode::Text);
    }

    let mut pieces = Vec::new();
    let mut current = String::new();
    let mut current_len = 0usize;

    for raw_item in items {
        let item = raw_item.trim();
        if item.is_empty() {
            continue;
        }
        let item_len = item.chars().count();

        if item_len > HARD_MAX_CHARS {
            if !current.trim().is_empty() {
                pieces.push(current.trim().to_string());
                current.clear();
                current_len = 0;
            }
            for part in split_with_preferences_with_overlap(item, SplitMode::Text, 0) {
                let part = part.trim();
                if !part.is_empty() {
                    pieces.push(part.to_string());
                }
            }
            continue;
        }

        let additional = if current.is_empty() {
            item_len
        } else {
            2 + item_len
        };
        if !current.is_empty() && current_len + additional > TARGET_CHARS {
            pieces.push(current.trim().to_string());
            current.clear();
            current_len = 0;
        }

        if !current.is_empty() {
            current.push_str("\n\n");
            current_len += 2;
        }
        current.push_str(item);
        current_len += item_len;
    }

    if !current.trim().is_empty() {
        pieces.push(current.trim().to_string());
    }

    if pieces.is_empty() {
        split_with_preferences(block, SplitMode::Text)
    } else {
        pieces
    }
}

fn extract_list_items(text: &str) -> Vec<String> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() > 1 {
        let mut items = Vec::new();
        let mut current = String::new();

        for line in lines {
            if line.trim().is_empty() {
                continue;
            }
            if is_list_start(line) {
                if !current.trim().is_empty() {
                    items.push(current.trim().to_string());
                    current.clear();
                }
                current.push_str(line.trim_end());
                continue;
            }
            if !current.is_empty() {
                current.push('\n');
                current.push_str(line.trim_end());
            }
        }

        if !current.trim().is_empty() {
            items.push(current.trim().to_string());
        }
        if items.len() >= 2 {
            return items;
        }
    }

    let trimmed = text.trim();
    if !(trimmed.starts_with("* ") || trimmed.starts_with("- ") || trimmed.starts_with("+ ")) {
        return Vec::new();
    }

    for delimiter in [" * [", " * `", " * <"] {
        if !trimmed.contains(delimiter) {
            continue;
        }
        let mut items = Vec::new();
        for (idx, part) in trimmed.split(delimiter).enumerate() {
            let value = part.trim();
            if value.is_empty() {
                continue;
            }
            if idx == 0 {
                if value.starts_with("* ") || value.starts_with("- ") || value.starts_with("+ ") {
                    items.push(value.to_string());
                } else {
                    items.push(format!("* {value}"));
                }
            } else {
                let rebuilt = match delimiter {
                    " * [" => format!("* [{value}"),
                    " * `" => format!("* `{value}"),
                    " * <" => format!("* <{value}"),
                    _ => format!("* {value}"),
                };
                items.push(rebuilt);
            }
        }
        if items.len() >= 2 {
            return items;
        }
    }

    Vec::new()
}

fn split_code_block(block: &str) -> Vec<String> {
    let lines: Vec<&str> = block.lines().collect();
    if lines.len() < 2 {
        return split_with_preferences(block, SplitMode::Line);
    }
    let open = lines[0].trim_end();
    let close = lines[lines.len() - 1].trim_end();
    if !open.trim_start().starts_with("```") || !close.trim_start().starts_with("```") {
        return split_with_preferences(block, SplitMode::Line);
    }

    let body = &lines[1..lines.len() - 1];
    if body.is_empty() {
        return vec![format!("{open}\n{close}")];
    }

    let overhead = open.chars().count() + close.chars().count() + 2;
    let target_body = TARGET_CHARS.saturating_sub(overhead).max(80);
    let hard_body = HARD_MAX_CHARS.saturating_sub(overhead).max(120);

    let mut pieces = Vec::new();
    let mut current: Vec<String> = Vec::new();
    let mut current_len = 0usize;

    for line in body {
        let line_len = line.chars().count() + 1;
        if line_len > hard_body {
            if !current.is_empty() {
                pieces.push(format!("{open}\n{}\n{close}", current.join("\n")));
                current.clear();
                current_len = 0;
            }
            for line_piece in split_with_preferences(line, SplitMode::Line) {
                pieces.push(format!("{open}\n{line_piece}\n{close}"));
            }
            continue;
        }

        if current_len + line_len > target_body && !current.is_empty() {
            pieces.push(format!("{open}\n{}\n{close}", current.join("\n")));
            current.clear();
            current_len = 0;
        }

        current.push((*line).to_string());
        current_len += line_len;
    }

    if !current.is_empty() {
        pieces.push(format!("{open}\n{}\n{close}", current.join("\n")));
    }

    pieces
}

fn split_table_block(block: &Block) -> Vec<BlockPiece> {
    let rows: Vec<String> = block.text.lines().map(|line| line.to_string()).collect();
    let table_info = block.table_info.clone().unwrap_or(TableBlockInfo {
        table_id: sha1_hex(&format!("table|{}", block.text)),
        table_ordinal: 0,
        column_headers: rows
            .first()
            .map(|row| parse_table_cells(row))
            .unwrap_or_default(),
    });

    let mut header_rows: Vec<String> = Vec::new();
    let mut data_rows: Vec<String> = Vec::new();

    let second_is_separator = rows.len() >= 2 && is_table_separator_line(&rows[1]);
    let has_later_separator = rows.iter().skip(2).any(|row| is_table_separator_line(row));

    if rows.len() >= 3 && second_is_separator {
        let first_row = &rows[0];
        let header_looks_like_data = first_row.chars().count() > 220
            || first_row.matches('`').count() >= 4
            || first_row.contains("<br/>");

        if !has_later_separator && !header_looks_like_data {
            // Standard markdown table: header + separator + data rows.
            header_rows = rows[..2].to_vec();
            data_rows = rows[2..].to_vec();
        } else {
            // Doxygen-style or malformed table where the first row is semantic data.
            data_rows = rows
                .into_iter()
                .filter(|row| !is_table_separator_line(row))
                .collect();
        }
    } else if rows.len() == 2 && is_table_separator_line(&rows[1]) {
        // Doxygen-style degenerate table: one data row followed by separator.
        data_rows.push(rows[0].clone());
    } else if rows.len() == 1 {
        // Single-row table without separator.
        data_rows.push(rows[0].clone());
    } else {
        // No explicit separator: treat each line as a data row.
        data_rows = rows;
    }

    let header_chars = header_rows
        .iter()
        .map(|r| r.chars().count() + 1)
        .sum::<usize>()
        + 1;
    let target_body = TARGET_CHARS.saturating_sub(header_chars).max(80);
    let hard_body = HARD_MAX_CHARS.saturating_sub(header_chars).max(120);

    let mut pieces: Vec<BlockPiece> = Vec::new();
    let mut current_rows: Vec<String> = Vec::new();
    let mut current_len = 0usize;

    for (row_ordinal, row) in data_rows.into_iter().enumerate() {
        let row_len = row.chars().count() + 1;
        if row_len > hard_body {
            if !current_rows.is_empty() {
                let mut piece_rows = header_rows.clone();
                piece_rows.extend(current_rows.clone());
                pieces.push(BlockPiece {
                    text: piece_rows.join("\n"),
                    table_part: None,
                });
                current_rows.clear();
                current_len = 0;
            }

            let cells = parse_table_cells(&row);
            if cells.is_empty() {
                for row_piece in split_with_preferences(&row, SplitMode::Line) {
                    let mut piece_rows = header_rows.clone();
                    piece_rows.push(row_piece);
                    pieces.push(BlockPiece {
                        text: piece_rows.join("\n"),
                        table_part: Some(TablePartMeta {
                            table_id: table_info.table_id.clone(),
                            table_ordinal: table_info.table_ordinal,
                            row_ordinal,
                            row_key: String::new(),
                            column_headers: table_info.column_headers.clone(),
                            cell_ordinal: 0,
                            cell_part_index: 0,
                            cell_part_total: 1,
                            is_table_continuation: false,
                        }),
                    });
                }
                continue;
            }

            let mut longest_idx = 0usize;
            let mut longest_len = 0usize;
            for (idx, cell) in cells.iter().enumerate() {
                let len = cell.chars().count();
                if len > longest_len {
                    longest_len = len;
                    longest_idx = idx;
                }
            }

            let row_key = sanitize_row_key(cells.first());
            let mut split_target = hard_body.saturating_sub(64).max(80);
            if split_target > TARGET_CHARS {
                split_target = TARGET_CHARS;
            }

            let cell_parts = split_table_cell_parts(&cells[longest_idx], split_target);
            let cell_part_total = cell_parts.len().max(1);

            for (part_idx, part) in cell_parts.into_iter().enumerate() {
                let mut part_cells = cells.clone();
                part_cells[longest_idx] = part;
                if let Some(first_cell) = part_cells.first_mut()
                    && cell_part_total > 1
                {
                    *first_cell = format!(
                        "{} (cont. {}/{})",
                        first_cell.trim(),
                        part_idx + 1,
                        cell_part_total
                    );
                }

                let mut row_text = build_table_row(&part_cells);
                if row_text.chars().count() > hard_body {
                    // If a row is still oversized after logical split, force split by line-aware strategy.
                    for overflow in split_with_preferences(&row_text, SplitMode::Line) {
                        let mut piece_rows = header_rows.clone();
                        piece_rows.push(overflow);
                        pieces.push(BlockPiece {
                            text: piece_rows.join("\n"),
                            table_part: Some(TablePartMeta {
                                table_id: table_info.table_id.clone(),
                                table_ordinal: table_info.table_ordinal,
                                row_ordinal,
                                row_key: row_key.clone(),
                                column_headers: table_info.column_headers.clone(),
                                cell_ordinal: longest_idx,
                                cell_part_index: part_idx,
                                cell_part_total,
                                is_table_continuation: cell_part_total > 1,
                            }),
                        });
                    }
                    continue;
                }

                let mut piece_rows = header_rows.clone();
                piece_rows.push(std::mem::take(&mut row_text));
                pieces.push(BlockPiece {
                    text: piece_rows.join("\n"),
                    table_part: Some(TablePartMeta {
                        table_id: table_info.table_id.clone(),
                        table_ordinal: table_info.table_ordinal,
                        row_ordinal,
                        row_key: row_key.clone(),
                        column_headers: table_info.column_headers.clone(),
                        cell_ordinal: longest_idx,
                        cell_part_index: part_idx,
                        cell_part_total,
                        is_table_continuation: cell_part_total > 1,
                    }),
                });
            }
            continue;
        }

        if current_len + row_len > target_body && !current_rows.is_empty() {
            let mut piece_rows = header_rows.clone();
            piece_rows.extend(current_rows.clone());
            pieces.push(BlockPiece {
                text: piece_rows.join("\n"),
                table_part: None,
            });
            current_rows.clear();
            current_len = 0;
        }

        current_rows.push(row);
        current_len += row_len;
    }

    if !current_rows.is_empty() {
        let mut piece_rows = header_rows;
        piece_rows.extend(current_rows);
        pieces.push(BlockPiece {
            text: piece_rows.join("\n"),
            table_part: None,
        });
    }

    pieces
}

fn parse_table_cells(row: &str) -> Vec<String> {
    let mut trimmed = row.trim();
    if trimmed.starts_with('|') {
        trimmed = &trimmed[1..];
    }
    if trimmed.ends_with('|') {
        trimmed = &trimmed[..trimmed.len().saturating_sub(1)];
    }
    trimmed
        .split('|')
        .map(|cell| cell.trim().to_string())
        .collect()
}

fn build_table_row(cells: &[String]) -> String {
    format!("| {} |", cells.join(" | "))
}

fn split_table_cell_parts(cell: &str, target_chars: usize) -> Vec<String> {
    let normalized_target = target_chars.max(80);
    if cell.chars().count() <= normalized_target {
        return vec![cell.to_string()];
    }

    if cell.contains("<br/>") {
        let tokens: Vec<&str> = cell.split("<br/>").collect();
        let mut parts = Vec::new();
        let mut current = String::new();

        for token in tokens {
            let token_trimmed = token.trim();
            if token_trimmed.is_empty() {
                continue;
            }

            let candidate = if current.is_empty() {
                token_trimmed.to_string()
            } else {
                format!("{current}<br/>{token_trimmed}")
            };

            if candidate.chars().count() <= normalized_target {
                current = candidate;
            } else {
                if !current.is_empty() {
                    parts.push(current);
                    current = token_trimmed.to_string();
                } else {
                    for p in split_with_preferences_with_overlap(token_trimmed, SplitMode::Text, 0)
                    {
                        parts.push(p);
                    }
                }
            }
        }

        if !current.is_empty() {
            parts.push(current);
        }

        if !parts.is_empty() {
            let cleaned = normalize_table_cell_parts(parts);
            if !cleaned.is_empty() {
                return cleaned;
            }
        }
    }

    let parts = split_with_preferences_with_overlap(cell, SplitMode::Text, 0);
    let cleaned = normalize_table_cell_parts(parts);
    if cleaned.is_empty() {
        vec![cell.trim().to_string()]
    } else {
        cleaned
    }
}

fn normalize_table_cell_parts(parts: Vec<String>) -> Vec<String> {
    let mut out = Vec::new();
    for (idx, part) in parts.into_iter().enumerate() {
        let mut cleaned = part.trim().to_string();
        if cleaned.is_empty() {
            continue;
        }
        if idx > 0 {
            while let Some(rest) = cleaned.strip_prefix("<br/>") {
                cleaned = rest.trim_start().to_string();
            }
            cleaned = cleaned
                .trim_start_matches(|ch: char| ch.is_whitespace() || matches!(ch, ',' | ';' | ':'))
                .trim_start()
                .to_string();
        }
        if !cleaned.is_empty() {
            out.push(cleaned);
        }
    }
    out
}

fn sanitize_row_key(first_cell: Option<&String>) -> String {
    let raw = first_cell.map(String::as_str).unwrap_or("").trim();
    if raw.is_empty() {
        return String::new();
    }

    let mut out = String::with_capacity(raw.len());
    let mut in_space = false;
    for ch in raw.chars() {
        if ch.is_whitespace() {
            if !in_space {
                out.push(' ');
                in_space = true;
            }
        } else {
            out.push(ch);
            in_space = false;
        }
    }
    let compact = out.trim();
    if compact.chars().count() <= 120 {
        compact.to_string()
    } else {
        compact.chars().take(120).collect()
    }
}

fn split_with_preferences(text: &str, mode: SplitMode) -> Vec<String> {
    split_with_preferences_with_overlap(text, mode, OVERLAP_CHARS)
}

fn split_with_preferences_with_overlap(
    text: &str,
    mode: SplitMode,
    overlap_chars: usize,
) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    if n <= HARD_MAX_CHARS {
        let trimmed = text.trim();
        return if trimmed.is_empty() {
            Vec::new()
        } else {
            vec![trimmed.to_string()]
        };
    }

    let mut out = Vec::new();
    let mut start = 0usize;

    while start < n {
        let remaining = n - start;
        if remaining <= HARD_MAX_CHARS {
            let piece: String = chars[start..n].iter().collect();
            let piece = piece.trim().to_string();
            if !piece.is_empty() {
                out.push(piece);
            }
            break;
        }

        let mut lo = start.saturating_add(TARGET_CHARS.saturating_sub(220));
        let mut hi = start
            .saturating_add(TARGET_CHARS.saturating_add(120))
            .min(start.saturating_add(HARD_MAX_CHARS))
            .min(n);

        lo = lo.max(start + 1).min(n);
        if hi <= lo {
            lo = (start + 1).min(n);
            hi = start.saturating_add(HARD_MAX_CHARS).min(n);
        }

        let cut = find_best_cut(&chars, start, lo, hi, mode)
            .or_else(|| {
                find_best_cut(
                    &chars,
                    start,
                    (start + 1).min(n),
                    start.saturating_add(HARD_MAX_CHARS).min(n),
                    SplitMode::Line,
                )
            })
            .unwrap_or_else(|| {
                start
                    .saturating_add(TARGET_CHARS)
                    .min(start.saturating_add(HARD_MAX_CHARS))
                    .min(n)
            });

        let cut = if cut <= start {
            (start + 1).min(n)
        } else {
            cut
        };
        let piece: String = chars[start..cut].iter().collect();
        let piece = piece.trim().to_string();
        if !piece.is_empty() {
            out.push(piece);
        }

        if cut >= n {
            break;
        }

        let overlap = overlap_chars.min(cut.saturating_sub(start).saturating_sub(1));
        let mut next_start = cut.saturating_sub(overlap);
        if next_start <= start {
            next_start = cut;
        }
        start = next_start;
    }

    out
}

fn find_best_cut(
    chars: &[char],
    start: usize,
    lo: usize,
    hi: usize,
    mode: SplitMode,
) -> Option<usize> {
    if lo >= hi || hi > chars.len() {
        return None;
    }

    match mode {
        SplitMode::Text => find_cut_by(chars, start, lo, hi, |prev, next| {
            is_sentence_end(prev) && next.map(char::is_whitespace).unwrap_or(true)
        })
        .or_else(|| {
            find_cut_by(chars, start, lo, hi, |prev, next| {
                matches!(prev, ';' | ':') && next.map(char::is_whitespace).unwrap_or(true)
            })
        })
        .or_else(|| {
            find_cut_by(chars, start, lo, hi, |prev, next| {
                prev == ',' && next.map(char::is_whitespace).unwrap_or(true)
            })
        })
        .or_else(|| find_cut_by(chars, start, lo, hi, |prev, _| prev == '\n'))
        .or_else(|| find_cut_by(chars, start, lo, hi, |prev, _| prev.is_whitespace())),
        SplitMode::Line => find_cut_by(chars, start, lo, hi, |prev, _| prev == '\n')
            .or_else(|| find_cut_by(chars, start, lo, hi, |prev, _| prev.is_whitespace())),
    }
}

fn find_cut_by<F>(
    chars: &[char],
    start: usize,
    lo: usize,
    hi: usize,
    mut predicate: F,
) -> Option<usize>
where
    F: FnMut(char, Option<char>) -> bool,
{
    for idx in (lo..=hi).rev() {
        if idx <= start || idx > chars.len() {
            continue;
        }
        let prev = chars[idx - 1];
        let next = chars.get(idx).copied();
        if predicate(prev, next) {
            return Some(idx);
        }
    }
    None
}

fn is_sentence_end(ch: char) -> bool {
    matches!(ch, '.' | '!' | '?' | '…')
}

fn merge_short_chunks(chunks: Vec<ChunkDraft>) -> Vec<ChunkDraft> {
    if chunks.len() <= 1 {
        return chunks;
    }

    let mut out: Vec<ChunkDraft> = Vec::new();
    for chunk in chunks {
        let chunk_len = chunk.content.chars().count();
        if chunk_len < MIN_CHARS
            && let Some(prev) = out.last_mut()
            && prev.table_part.is_none()
            && chunk.table_part.is_none()
            && prev.content.chars().count() + 2 + chunk_len <= HARD_MAX_CHARS
        {
            prev.content.push_str("\n\n");
            prev.content.push_str(chunk.content.trim());
            prev.has_code |= chunk.has_code;
            prev.has_table |= chunk.has_table;
            prev.has_list |= chunk.has_list;
            continue;
        }
        out.push(chunk);
    }

    if out.len() >= 2 {
        let mut idx = 0usize;
        while idx + 1 < out.len() {
            if out[idx].content.chars().count() < MIN_CHARS {
                let can_merge =
                    out[idx].content.chars().count() + 2 + out[idx + 1].content.chars().count()
                        <= HARD_MAX_CHARS;
                if can_merge && out[idx].table_part.is_none() && out[idx + 1].table_part.is_none() {
                    let short = out.remove(idx);
                    let next = &mut out[idx];
                    next.content = format!("{}\n\n{}", short.content.trim(), next.content.trim());
                    next.has_code |= short.has_code;
                    next.has_table |= short.has_table;
                    next.has_list |= short.has_list;
                    continue;
                }
            }
            idx += 1;
        }
    }

    if out.len() >= 2 {
        let last_idx = out.len() - 1;
        if out[last_idx].content.chars().count() < MIN_CHARS {
            let can_merge = out[last_idx - 1].content.chars().count()
                + 2
                + out[last_idx].content.chars().count()
                <= HARD_MAX_CHARS;
            if can_merge
                && out[last_idx - 1].table_part.is_none()
                && out[last_idx].table_part.is_none()
            {
                let last = out.pop().expect("last chunk must exist");
                let prev = out
                    .last_mut()
                    .expect("previous chunk must exist when len >= 1");
                prev.content.push_str("\n\n");
                prev.content.push_str(last.content.trim());
                prev.has_code |= last.has_code;
                prev.has_table |= last.has_table;
                prev.has_list |= last.has_list;
            }
        }
    }

    out
}

fn build_source_url(bucket: &str, file_name: &str) -> Result<String> {
    if !file_name.ends_with(".md") {
        return Err(anyhow!("source file is not markdown: {file_name}"));
    }

    let stem = file_name.trim_end_matches(".md");
    let segments = if stem == "index" {
        Vec::new()
    } else {
        decode_flat_segments(stem)
    };

    let path = if bucket == "default" {
        if segments.is_empty() {
            "/doc".to_string()
        } else {
            format!("/doc/{}", segments.join("/"))
        }
    } else if segments.is_empty() {
        format!("/doc/{bucket}")
    } else {
        format!("/doc/{bucket}/{}", segments.join("/"))
    };

    Ok(format!("{BASE_DOCS_URL}{path}"))
}

fn make_source_url_with_anchor(source_url: &str, anchor: Option<&str>) -> String {
    match anchor {
        Some(raw) if !raw.trim().is_empty() => format!("{source_url}#{raw}"),
        _ => source_url.to_string(),
    }
}

fn decode_flat_segments(stem: &str) -> Vec<String> {
    let chars: Vec<char> = stem.chars().collect();
    let mut segments = Vec::new();
    let mut current = String::new();
    let mut i = 0usize;

    while i < chars.len() {
        if i + 3 < chars.len()
            && chars[i] == '_'
            && chars[i + 1] == '_'
            && chars[i + 2] == '_'
            && chars[i + 3] == '_'
        {
            current.push_str("__");
            i += 4;
            continue;
        }

        if i + 1 < chars.len() && chars[i] == '_' && chars[i + 1] == '_' {
            segments.push(current);
            current = String::new();
            i += 2;
            continue;
        }

        current.push(chars[i]);
        i += 1;
    }

    segments.push(current);
    segments
}

fn sha1_hex(input: &str) -> String {
    use sha1::{Digest, Sha1};
    let mut hasher = Sha1::new();
    hasher.update(input.as_bytes());
    let bytes = hasher.finalize();
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn shorten_for_progress(value: &str, max_len: usize) -> String {
    if max_len <= 3 {
        return value.to_string();
    }
    let char_count = value.chars().count();
    if char_count <= max_len {
        return value.to_string();
    }
    let keep = max_len - 3;
    let tail: String = value.chars().skip(char_count - keep).collect();
    format!("...{tail}")
}

fn is_version_segment(segment: &str) -> bool {
    if !segment.contains('.') {
        return false;
    }
    segment
        .split('.')
        .all(|part| !part.is_empty() && part.chars().all(|ch| ch.is_ascii_digit()))
}

fn now_rfc3339() -> String {
    Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_flat_segments_unescapes_double_underscore() {
        let segments = decode_flat_segments("software_development__foo____bar__baz");
        assert_eq!(
            segments,
            vec![
                "software_development".to_string(),
                "foo__bar".to_string(),
                "baz".to_string()
            ]
        );
    }

    #[test]
    fn build_source_url_for_default_and_versioned() {
        let default_url =
            build_source_url("default", "software_development__guides__first_steps.md")
                .expect("default url must be built");
        assert_eq!(
            default_url,
            "https://developer.auroraos.ru/doc/software_development/guides/first_steps"
        );

        let version_url = build_source_url("5.2.0", "software_development__guides__first_steps.md")
            .expect("version url must be built");
        assert_eq!(
            version_url,
            "https://developer.auroraos.ru/doc/5.2.0/software_development/guides/first_steps"
        );
    }

    #[test]
    fn build_source_url_for_index_documents() {
        let default_index = build_source_url("default", "index.md").expect("default index url");
        assert_eq!(default_index, "https://developer.auroraos.ru/doc");

        let version_index = build_source_url("5.2.1", "index.md").expect("version index url");
        assert_eq!(version_index, "https://developer.auroraos.ru/doc/5.2.1");
    }

    #[test]
    fn parse_heading_extracts_anchor() {
        let parsed = parse_heading_line("## Methods {#methods}");
        assert_eq!(
            parsed,
            Some((2usize, "Methods".to_string(), Some("methods".to_string())))
        );
    }

    #[test]
    fn split_text_prefers_sentence_boundary() {
        let sentence = "This sentence should stay whole. ";
        let text = sentence.repeat(150);
        let pieces = split_with_preferences(&text, SplitMode::Text);
        assert!(pieces.len() > 1);
        assert!(pieces[0].ends_with('.'));
        assert!(pieces.iter().all(|p| p.chars().count() <= HARD_MAX_CHARS));
    }

    #[test]
    fn split_code_block_keeps_fences() {
        let body = (0..400)
            .map(|i| format!("let x{i} = {i};"))
            .collect::<Vec<_>>()
            .join("\n");
        let code = format!("```rust\n{body}\n```");

        let pieces = split_code_block(&code);
        assert!(pieces.len() > 1);
        for piece in pieces {
            assert!(piece.starts_with("```rust\n"));
            assert!(piece.ends_with("\n```"));
        }
    }

    #[test]
    fn split_table_block_repeats_header() {
        let mut lines = vec!["| A | B |".to_string(), "| --- | --- |".to_string()];
        for i in 0..500 {
            lines.push(format!("| row{i} | value{i} |"));
        }
        let table = lines.join("\n");
        let block = Block {
            kind: BlockKind::Table,
            text: table,
            table_info: Some(TableBlockInfo {
                table_id: "table-id".to_string(),
                table_ordinal: 0,
                column_headers: vec!["A".to_string(), "B".to_string()],
            }),
        };
        let pieces = split_table_block(&block);

        assert!(pieces.len() > 1);
        for piece in pieces {
            assert!(
                piece
                    .text
                    .lines()
                    .next()
                    .unwrap_or_default()
                    .starts_with("| A | B |")
            );
            assert!(piece.text.contains("| --- | --- |"));
        }
    }

    #[test]
    fn split_table_block_oversized_row_has_part_metadata() {
        let long_cell = (0..1500)
            .map(|i| format!("ERR_{i} = -{i}"))
            .collect::<Vec<_>>()
            .join("<br/>");
        let table = format!("| Kind | Value |\n| --- | --- |\n| enum | {long_cell} |");
        let block = Block {
            kind: BlockKind::Table,
            text: table,
            table_info: Some(TableBlockInfo {
                table_id: "table-overflow".to_string(),
                table_ordinal: 7,
                column_headers: vec!["Kind".to_string(), "Value".to_string()],
            }),
        };

        let pieces = split_table_block(&block);
        assert!(pieces.len() > 1);
        assert!(
            pieces
                .iter()
                .all(|p| p.text.chars().count() <= HARD_MAX_CHARS)
        );
        assert!(pieces.iter().all(|p| p.table_part.is_some()));
        assert!(pieces.iter().all(|p| p.text.contains("| --- | --- |")));

        let totals: Vec<usize> = pieces
            .iter()
            .map(|p| {
                p.table_part
                    .as_ref()
                    .expect("table part meta must be set")
                    .cell_part_total
            })
            .collect();
        let max_total = totals.into_iter().max().unwrap_or(0);
        assert!(max_total > 1);
    }

    #[test]
    fn split_list_block_members_style_keeps_item_boundaries() {
        let items = (0..320)
            .map(|i| format!("[item{i}](https://example.com/{i}) : value{i}"))
            .collect::<Vec<_>>();
        let list = format!("* {}", items.join(" * "));
        let block = Block {
            kind: BlockKind::List,
            text: list,
            table_info: None,
        };

        let pieces = split_oversized_block(&block);
        assert!(pieces.len() > 1);
        assert!(
            pieces
                .iter()
                .all(|p| p.text.chars().count() <= HARD_MAX_CHARS)
        );
        assert!(pieces.iter().all(|p| p.text.trim_start().starts_with('*')));
    }

    #[test]
    fn split_table_cell_parts_trim_leading_separators_on_continuation() {
        let cell = (0..300)
            .map(|i| format!("entry{i}"))
            .collect::<Vec<_>>()
            .join(" , ");
        let parts = split_table_cell_parts(&cell, 120);
        assert!(parts.len() > 1);
        assert!(
            parts
                .iter()
                .skip(1)
                .all(|p| !p.starts_with(',') && !p.starts_with(';') && !p.starts_with(':'))
        );
    }
}
