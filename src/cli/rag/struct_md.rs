use super::RagStructArgs;
use crate::config::AppConfig;
use anyhow::{Context, Result, anyhow};
use chrono::{SecondsFormat, Utc};
use ego_tree::NodeRef;
use indicatif::{ProgressBar, ProgressStyle};
use scraper::{ElementRef, Html, Selector, node::Node};
use serde::Serialize;
use serde_json::Value;
use std::{
    fs,
    path::{Path, PathBuf},
};

const HTML_ROOT_DIRNAME: &str = "html_data";
const MD_ROOT_DIRNAME: &str = "md_data";
const DOC_ROOT: &str = "doc";
const INDEX_HTML: &str = "index.html";
const INDEX_MD: &str = "index.md";
const HIDDEN_ANCHOR_STYLE_PART: &str = "position: absolute";

macro_rules! vprintln {
    ($verbose:expr, $($arg:tt)*) => {
        if $verbose {
            println!($($arg)*);
        }
    };
}

#[derive(Debug, Serialize)]
struct StructManifest {
    started_at: String,
    finished_at: String,
    input_root: String,
    output_root: String,
    total_inputs: usize,
    converted_total: usize,
    errors_total: usize,
    records: Vec<StructManifestRecord>,
}

#[derive(Debug, Serialize)]
struct StructManifestRecord {
    input_html: String,
    bucket: String,
    output_md: Option<String>,
    status: &'static str,
    chars: Option<usize>,
    error: Option<String>,
}

#[derive(Debug)]
struct OutputTarget {
    bucket: String,
    file_name: String,
}

pub fn run(args: RagStructArgs) -> Result<()> {
    let verbose = args.verbose;
    let cfg = AppConfig::load()?;
    let data_root = cfg.data_root()?;
    let input_root = data_root.join(HTML_ROOT_DIRNAME);
    let output_root = data_root.join(MD_ROOT_DIRNAME);
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

    vprintln!(verbose, "[struct] input: {}", input_root.display());
    vprintln!(verbose, "[struct] output: {}", output_root.display());

    let files = discover_input_files(&input_root)?;
    let total_inputs = files.len();

    let started_at = now_rfc3339();
    let mut converted_total = 0usize;
    let mut errors_total = 0usize;
    let mut records: Vec<StructManifestRecord> = Vec::with_capacity(total_inputs);
    let progress = if verbose {
        let pb = ProgressBar::new(total_inputs as u64);
        let style =
            ProgressStyle::with_template("[struct] [{bar:28.green/white}] {pos}/{len}: {msg}")
                .expect("progress template must be valid")
                .progress_chars("=> ");
        pb.set_style(style);
        Some(pb)
    } else {
        None
    };

    for rel_path in &files {
        let input_path = input_root.join(rel_path);
        let output_target = map_html_rel_to_output(rel_path)?;
        let output_bucket_dir = output_root.join(&output_target.bucket);
        fs::create_dir_all(&output_bucket_dir).with_context(|| {
            format!(
                "failed to create bucket directory: {}",
                output_bucket_dir.display()
            )
        })?;
        let output_path = output_bucket_dir.join(&output_target.file_name);

        let record = match convert_one_file(
            &input_path,
            rel_path,
            &output_root,
            &output_path,
            &output_target,
        ) {
            Ok(record) => {
                converted_total += 1;
                record
            }
            Err(err) => {
                errors_total += 1;
                if let Some(pb) = &progress {
                    let rel = rel_path.display().to_string();
                    pb.suspend(|| eprintln!("[struct][warn] {}: {}", rel, err));
                }
                StructManifestRecord {
                    input_html: rel_path.to_string_lossy().replace('\\', "/"),
                    bucket: output_target.bucket.clone(),
                    output_md: None,
                    status: "error",
                    chars: None,
                    error: Some(err.to_string()),
                }
            }
        };
        records.push(record);

        if let Some(pb) = &progress {
            let current_display = rel_path.to_string_lossy().replace('\\', "/");
            pb.set_message(shorten_for_progress(&current_display, 100));
            pb.inc(1);
        }
    }

    if let Some(pb) = progress {
        pb.finish_and_clear();
    }

    let finished_at = now_rfc3339();
    let manifest = StructManifest {
        started_at,
        finished_at,
        input_root: input_root.display().to_string(),
        output_root: output_root.display().to_string(),
        total_inputs,
        converted_total,
        errors_total,
        records,
    };

    let manifest_path = output_root.join("manifest.json");
    let manifest_json =
        serde_json::to_vec_pretty(&manifest).context("failed to serialize struct manifest")?;
    fs::write(&manifest_path, manifest_json)
        .with_context(|| format!("failed to write manifest: {}", manifest_path.display()))?;

    vprintln!(
        verbose,
        "[struct] finished: total={}, converted={}, errors={}, manifest={}",
        manifest.total_inputs,
        manifest.converted_total,
        manifest.errors_total,
        manifest_path.display()
    );
    println!(
        "{} documents converted and stored in {}",
        converted_total,
        output_root.display()
    );

    Ok(())
}

fn discover_input_files(input_root: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    walk_collect_index_html(input_root, input_root, &mut files)?;
    files.sort();
    Ok(files)
}

fn walk_collect_index_html(root: &Path, current: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(current)
        .with_context(|| format!("failed to read directory: {}", current.display()))?
    {
        let entry = entry
            .with_context(|| format!("failed to read directory entry in {}", current.display()))?;
        let path = entry.path();
        let file_type = entry.file_type().with_context(|| {
            format!(
                "failed to read file type for directory entry: {}",
                path.display()
            )
        })?;
        if file_type.is_dir() {
            walk_collect_index_html(root, &path, out)?;
            continue;
        }
        if !file_type.is_file() {
            continue;
        }

        if entry.file_name().to_string_lossy() != INDEX_HTML {
            continue;
        }

        let rel = path
            .strip_prefix(root)
            .with_context(|| format!("failed to build relative path for {}", path.display()))?;

        let mut comps = rel.components();
        let first = comps.next();
        if first.map(|c| c.as_os_str().to_string_lossy().to_string()) != Some(DOC_ROOT.to_string())
        {
            continue;
        }

        out.push(rel.to_path_buf());
    }
    Ok(())
}

fn map_html_rel_to_output(rel_html: &Path) -> Result<OutputTarget> {
    let components: Vec<String> = rel_html
        .components()
        .map(|comp| comp.as_os_str().to_string_lossy().to_string())
        .collect();

    if components.len() < 2 {
        return Err(anyhow!(
            "unexpected html path shape: {}",
            rel_html.display()
        ));
    }
    if components.first().map(String::as_str) != Some(DOC_ROOT) {
        return Err(anyhow!(
            "html path must start with 'doc': {}",
            rel_html.display()
        ));
    }
    if components.last().map(String::as_str) != Some(INDEX_HTML) {
        return Err(anyhow!(
            "html file must be index.html: {}",
            rel_html.display()
        ));
    }

    let mut content_segments = components[1..components.len() - 1].to_vec();
    let bucket = if content_segments
        .first()
        .map(|s| is_version_segment(s))
        .unwrap_or(false)
    {
        content_segments.remove(0)
    } else {
        "default".to_string()
    };

    let file_name = if content_segments.is_empty() {
        INDEX_MD.to_string()
    } else {
        let flat = content_segments
            .iter()
            .map(|segment| sanitize_flat_segment(segment))
            .collect::<Vec<_>>()
            .join("__");
        format!("{flat}.md")
    };

    Ok(OutputTarget { bucket, file_name })
}

fn sanitize_flat_segment(segment: &str) -> String {
    let escaped = segment.replace("__", "____");
    let mut out = String::with_capacity(escaped.len());
    for ch in escaped.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() { "_".to_string() } else { out }
}

fn convert_one_file(
    input_path: &Path,
    rel_input_path: &Path,
    output_root: &Path,
    output_path: &Path,
    target: &OutputTarget,
) -> Result<StructManifestRecord> {
    let raw_html = fs::read_to_string(input_path)
        .with_context(|| format!("failed to read html file: {}", input_path.display()))?;
    let article_html = extract_article_html(&raw_html);
    let markdown = convert_article_html_to_markdown(&article_html);

    fs::write(output_path, markdown.as_bytes())
        .with_context(|| format!("failed to write markdown file: {}", output_path.display()))?;

    Ok(StructManifestRecord {
        input_html: rel_input_path.to_string_lossy().replace('\\', "/"),
        bucket: target.bucket.clone(),
        output_md: Some(output_target_path_display(output_root, output_path)),
        status: "ok",
        chars: Some(markdown.chars().count()),
        error: None,
    })
}

fn extract_article_html(raw_html: &str) -> String {
    let document = Html::parse_document(raw_html);

    if let Ok(selector) = Selector::parse(r#"script#__NEXT_DATA__"#)
        && let Some(script_node) = document.select(&selector).next()
    {
        let raw_json = script_node.text().collect::<String>();
        if let Ok(value) = serde_json::from_str::<Value>(&raw_json)
            && let Some(content) = value
                .pointer("/props/pageProps/content")
                .and_then(Value::as_str)
            && !content.trim().is_empty()
        {
            return content.to_string();
        }
    }

    if let Ok(selector) = Selector::parse("article .prose")
        && let Some(node) = document.select(&selector).next()
    {
        let html = node.inner_html();
        if !html.trim().is_empty() {
            return html;
        }
    }

    if let Ok(selector) = Selector::parse("body")
        && let Some(node) = document.select(&selector).next()
    {
        let html = node.inner_html();
        if !html.trim().is_empty() {
            return html;
        }
    }

    raw_html.to_string()
}

fn convert_article_html_to_markdown(article_html: &str) -> String {
    let wrapped = format!("<html><body>{article_html}</body></html>");
    let document = Html::parse_document(&wrapped);
    let mut md = String::new();

    let body_selector = Selector::parse("body").expect("body selector must be valid");
    if let Some(body) = document.select(&body_selector).next() {
        for child in body.children() {
            md.push_str(&render_block_node(child, 0));
        }
    }

    normalize_markdown(md)
}

fn render_block_node(node: NodeRef<'_, Node>, depth: usize) -> String {
    match node.value() {
        Node::Text(text) => {
            let normalized = normalize_ws(text);
            let trimmed = normalized.trim();
            if trimmed.is_empty() {
                String::new()
            } else {
                format!("{trimmed}\n\n")
            }
        }
        Node::Element(_) => ElementRef::wrap(node)
            .map(|el| render_block_element(el, depth))
            .unwrap_or_default(),
        _ => String::new(),
    }
}

fn render_block_element(element: ElementRef<'_>, depth: usize) -> String {
    let name = element.value().name();
    match name {
        "html" | "body" | "section" | "article" | "div" => {
            let mut out = String::new();
            for child in element.children() {
                out.push_str(&render_block_node(child, depth));
            }
            out
        }
        "h1" | "h2" | "h3" | "h4" | "h5" | "h6" => {
            let level = name[1..].parse::<usize>().unwrap_or(1);
            let text = render_inline_children(element).trim().to_string();
            if text.is_empty() {
                return String::new();
            }
            let suffix = heading_anchor(element)
                .map(|anchor| format!(" {{#{anchor}}}"))
                .unwrap_or_default();
            format!("{} {}{}\n\n", "#".repeat(level), text, suffix)
        }
        "p" => {
            let text = render_inline_children(element).trim().to_string();
            if text.is_empty() {
                String::new()
            } else {
                format!("{text}\n\n")
            }
        }
        "ul" | "ol" => render_list(element, depth),
        "pre" => render_pre(element),
        "blockquote" => render_blockquote(element, depth),
        "table" => render_table(element),
        "hr" => "---\n\n".to_string(),
        "dl" => render_definition_list(element),
        _ => {
            let text = render_inline_children(element).trim().to_string();
            if text.is_empty() {
                String::new()
            } else {
                format!("{text}\n\n")
            }
        }
    }
}

fn render_inline_children(element: ElementRef<'_>) -> String {
    let mut out = String::new();
    for child in element.children() {
        out.push_str(&render_inline_node(child));
    }
    out
}

fn render_inline_node(node: NodeRef<'_, Node>) -> String {
    match node.value() {
        Node::Text(text) => normalize_ws(text),
        Node::Element(_) => ElementRef::wrap(node)
            .map(render_inline_element)
            .unwrap_or_default(),
        _ => String::new(),
    }
}

fn render_inline_element(element: ElementRef<'_>) -> String {
    match element.value().name() {
        "a" => {
            let href = element.value().attr("href").unwrap_or("").trim();
            let mut text = render_inline_children(element).trim().to_string();
            if text.is_empty() {
                text = href.to_string();
            }
            format!("[{text}]({href})")
        }
        "img" => {
            let src = element.value().attr("src").unwrap_or("").trim();
            let alt = element.value().attr("alt").unwrap_or("");
            format!("![{alt}]({src})")
        }
        "strong" | "b" => {
            let content = render_inline_children(element).trim().to_string();
            format!("**{content}**")
        }
        "em" | "i" => {
            let content = render_inline_children(element).trim().to_string();
            format!("*{content}*")
        }
        "code" => wrap_inline_code(&collect_text(element)),
        "br" => "<br/>".to_string(),
        "span" => {
            let is_hidden_anchor = element.value().attr("id").is_some()
                && element
                    .value()
                    .attr("style")
                    .map(|style| {
                        style
                            .to_ascii_lowercase()
                            .contains(HIDDEN_ANCHOR_STYLE_PART)
                    })
                    .unwrap_or(false);
            if is_hidden_anchor {
                String::new()
            } else {
                render_inline_children(element)
            }
        }
        _ => render_inline_children(element),
    }
}

fn heading_anchor(element: ElementRef<'_>) -> Option<String> {
    for child in element.children() {
        match child.value() {
            Node::Text(text) => {
                if normalize_ws(text).trim().is_empty() {
                    continue;
                }
                break;
            }
            Node::Element(_) => {
                let Some(child_el) = ElementRef::wrap(child) else {
                    break;
                };
                if child_el.value().name() == "span"
                    && let Some(id) = child_el.value().attr("id")
                {
                    let style = child_el
                        .value()
                        .attr("style")
                        .unwrap_or("")
                        .to_ascii_lowercase();
                    if style.contains(HIDDEN_ANCHOR_STYLE_PART) {
                        return Some(id.to_string());
                    }
                }
                break;
            }
            _ => break,
        }
    }
    None
}

fn render_list(element: ElementRef<'_>, depth: usize) -> String {
    let ordered = element.value().name() == "ol";
    let mut out = String::new();
    let mut index = 1usize;

    for child in element.children() {
        let Some(li) = ElementRef::wrap(child) else {
            continue;
        };
        if li.value().name() != "li" {
            continue;
        }
        out.push_str(&render_list_item(li, depth, ordered, index));
        index += 1;
    }

    out.push('\n');
    out
}

fn render_list_item(item: ElementRef<'_>, depth: usize, ordered: bool, index: usize) -> String {
    let indent = "  ".repeat(depth);
    let bullet = if ordered {
        format!("{index}. ")
    } else {
        "* ".to_string()
    };

    let mut main_parts: Vec<String> = Vec::new();
    let mut nested_parts: Vec<String> = Vec::new();

    for child in item.children() {
        match child.value() {
            Node::Text(text) => {
                let normalized = normalize_ws(text).trim().to_string();
                if !normalized.is_empty() {
                    main_parts.push(normalized);
                }
            }
            Node::Element(_) => {
                let Some(child_el) = ElementRef::wrap(child) else {
                    continue;
                };
                match child_el.value().name() {
                    "ul" | "ol" => nested_parts.push(render_list(child_el, depth + 1)),
                    "p" => {
                        let text = render_inline_children(child_el).trim().to_string();
                        if !text.is_empty() {
                            main_parts.push(text);
                        }
                    }
                    "pre" | "table" => nested_parts.push(render_block_element(child_el, depth + 1)),
                    _ => {
                        let text = render_inline_element(child_el).trim().to_string();
                        if !text.is_empty() {
                            main_parts.push(text);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    let main_text = main_parts.join(" ").trim().to_string();
    let mut line = format!("{indent}{bullet}{main_text}\n");
    for nested in nested_parts {
        line.push_str(&nested);
    }
    line
}

fn render_pre(element: ElementRef<'_>) -> String {
    let mut language = String::new();
    let mut content = String::new();

    let code_selector = Selector::parse("code").expect("code selector must be valid");
    if let Some(code) = element.select(&code_selector).next() {
        if let Some(class_attr) = code.value().attr("class") {
            language = parse_language(class_attr);
        }
        content = collect_text(code);
    }

    if content.is_empty() {
        content = collect_text(element);
    }

    let content = content.trim_end_matches('\n');
    let fence = if content.contains("```") {
        "````"
    } else {
        "```"
    };
    format!("{fence}{language}\n{content}\n{fence}\n\n")
}

fn render_blockquote(element: ElementRef<'_>, depth: usize) -> String {
    let mut raw = String::new();
    for child in element.children() {
        raw.push_str(&render_block_node(child, depth));
    }
    let normalized = collapse_blank_lines(&raw);
    let trimmed = normalized.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    let lines = trimmed
        .lines()
        .map(|line| {
            if line.is_empty() {
                ">".to_string()
            } else {
                format!("> {line}")
            }
        })
        .collect::<Vec<_>>();
    format!("{}\n\n", lines.join("\n"))
}

fn render_table(table: ElementRef<'_>) -> String {
    let tr_selector = Selector::parse("tr").expect("tr selector must be valid");
    let mut rows: Vec<Vec<ElementRef<'_>>> = Vec::new();

    for tr in table.select(&tr_selector) {
        let mut cells: Vec<ElementRef<'_>> = Vec::new();
        for child in tr.children() {
            let Some(cell) = ElementRef::wrap(child) else {
                continue;
            };
            if matches!(cell.value().name(), "th" | "td") {
                cells.push(cell);
            }
        }
        if !cells.is_empty() {
            rows.push(cells);
        }
    }

    if rows.is_empty() {
        return String::new();
    }

    let header = &rows[0];
    let header_vals = header
        .iter()
        .map(|cell| render_table_cell(*cell))
        .collect::<Vec<_>>();
    let aligns = header
        .iter()
        .map(|cell| parse_cell_align(*cell))
        .collect::<Vec<_>>();
    let col_count = header_vals.len();

    let mut lines = Vec::new();
    lines.push(format!("| {} |", header_vals.join(" | ")));

    let align_line = aligns
        .iter()
        .map(|align| match align.as_str() {
            "left" => ":--",
            "right" => "--:",
            "center" => ":-:",
            _ => "---",
        })
        .collect::<Vec<_>>();
    lines.push(format!("| {} |", align_line.join(" | ")));

    for row in rows.iter().skip(1) {
        let mut vals = row
            .iter()
            .map(|cell| render_table_cell(*cell))
            .collect::<Vec<_>>();
        if vals.len() < col_count {
            vals.extend(std::iter::repeat_n(String::new(), col_count - vals.len()));
        } else if vals.len() > col_count {
            vals.truncate(col_count);
        }
        lines.push(format!("| {} |", vals.join(" | ")));
    }

    format!("{}\n\n", lines.join("\n"))
}

fn render_table_cell(cell: ElementRef<'_>) -> String {
    let mut parts: Vec<String> = Vec::new();

    for child in cell.children() {
        match child.value() {
            Node::Text(text) => {
                let normalized = normalize_ws(text).trim().to_string();
                if !normalized.is_empty() {
                    parts.push(normalized);
                }
            }
            Node::Element(_) => {
                let Some(child_el) = ElementRef::wrap(child) else {
                    continue;
                };

                match child_el.value().name() {
                    "br" => parts.push("<br/>".to_string()),
                    "p" | "div" => {
                        let text = render_inline_children(child_el).trim().to_string();
                        if !text.is_empty() {
                            if !parts.is_empty()
                                && parts.last().map(String::as_str) != Some("<br/>")
                            {
                                parts.push("<br/>".to_string());
                            }
                            parts.push(text);
                        }
                    }
                    "ul" | "ol" => {
                        let mut items = Vec::new();
                        for li_node in child_el.children() {
                            let Some(li) = ElementRef::wrap(li_node) else {
                                continue;
                            };
                            if li.value().name() != "li" {
                                continue;
                            }
                            let text = render_inline_children(li).trim().to_string();
                            if !text.is_empty() {
                                items.push(format!("* {text}"));
                            }
                        }

                        if !items.is_empty() {
                            if !parts.is_empty()
                                && parts.last().map(String::as_str) != Some("<br/>")
                            {
                                parts.push("<br/>".to_string());
                            }
                            parts.extend(items);
                        }
                    }
                    _ => {
                        let text = render_inline_element(child_el).trim().to_string();
                        if !text.is_empty() {
                            parts.push(text);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    let mut out = String::new();
    for part in parts {
        if part == "<br/>" {
            out = out.trim_end().to_string();
            out.push_str("<br/>");
            continue;
        }

        if !out.is_empty() && !out.ends_with("<br/>") {
            out.push(' ');
        }
        out.push_str(&part);
    }

    out.trim().replace('|', r"\|")
}

fn parse_cell_align(cell: ElementRef<'_>) -> String {
    if let Some(align) = cell.value().attr("align") {
        let align_lower = align.to_ascii_lowercase();
        if matches!(align_lower.as_str(), "left" | "right" | "center") {
            return align_lower;
        }
    }

    if let Some(style) = cell.value().attr("style") {
        let style_lower = style.to_ascii_lowercase();
        for align in ["left", "right", "center"] {
            if style_lower.contains(&format!("text-align:{align}"))
                || style_lower.contains(&format!("text-align: {align}"))
            {
                return align.to_string();
            }
        }
    }

    String::new()
}

fn render_definition_list(dl: ElementRef<'_>) -> String {
    let mut nodes: Vec<ElementRef<'_>> = Vec::new();
    for child in dl.children() {
        let Some(el) = ElementRef::wrap(child) else {
            continue;
        };
        if matches!(el.value().name(), "dt" | "dd") {
            nodes.push(el);
        }
    }

    if nodes.is_empty() {
        return String::new();
    }

    let mut lines: Vec<String> = Vec::new();
    let mut idx = 0usize;
    while idx < nodes.len() {
        if nodes[idx].value().name() != "dt" {
            idx += 1;
            continue;
        }

        let mut term = render_inline_children(nodes[idx]).trim().to_string();
        if let Some(id) = nodes[idx].value().attr("id") {
            if !term.is_empty() {
                term.push_str(&format!(" {{#{id}}}"));
            }
        }
        idx += 1;

        let mut defs: Vec<String> = Vec::new();
        while idx < nodes.len() && nodes[idx].value().name() == "dd" {
            let definition = render_inline_children(nodes[idx]).trim().to_string();
            if !definition.is_empty() {
                defs.push(definition);
            }
            idx += 1;
        }

        if !term.is_empty() {
            lines.push(term);
            for definition in defs {
                lines.push(format!(": {definition}"));
            }
            lines.push(String::new());
        }
    }

    if lines.is_empty() {
        String::new()
    } else {
        format!("{}\n\n", lines.join("\n").trim())
    }
}

fn parse_language(class_attr: &str) -> String {
    class_attr
        .split_whitespace()
        .find_map(|class_name| class_name.strip_prefix("language-"))
        .unwrap_or("")
        .to_string()
}

fn wrap_inline_code(text: &str) -> String {
    let compact = text.replace('\n', " ").trim().to_string();
    if compact.is_empty() {
        return "``".to_string();
    }

    let mut max_ticks = 0usize;
    let mut current_ticks = 0usize;
    for ch in compact.chars() {
        if ch == '`' {
            current_ticks += 1;
            max_ticks = max_ticks.max(current_ticks);
        } else {
            current_ticks = 0;
        }
    }

    let delim = "`".repeat(max_ticks + 1);
    if compact.starts_with('`') || compact.ends_with('`') {
        format!("{delim} {compact} {delim}")
    } else {
        format!("{delim}{compact}{delim}")
    }
}

fn collect_text(element: ElementRef<'_>) -> String {
    let mut out = String::new();
    for child in element.children() {
        collect_text_recursive(child, &mut out);
    }
    out
}

fn collect_text_recursive(node: NodeRef<'_, Node>, out: &mut String) {
    match node.value() {
        Node::Text(text) => out.push_str(text),
        _ => {
            for child in node.children() {
                collect_text_recursive(child, out);
            }
        }
    }
}

fn normalize_ws(input: &str) -> String {
    let normalized = input.replace('\u{00A0}', " ");
    let mut out = String::new();
    let mut in_space = false;
    for ch in normalized.chars() {
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
    out
}

fn normalize_markdown(md: String) -> String {
    let normalized = md.replace("\r\n", "\n");
    let collapsed = collapse_blank_lines(&normalized);
    let trimmed = collapsed.trim();
    if trimmed.is_empty() {
        String::new()
    } else {
        format!("{trimmed}\n")
    }
}

fn collapse_blank_lines(text: &str) -> String {
    let mut out = String::new();
    let mut blank_run = 0usize;

    for line in text.lines() {
        if line.trim().is_empty() {
            blank_run += 1;
            if blank_run <= 2 {
                out.push('\n');
            }
            continue;
        }

        blank_run = 0;
        out.push_str(line);
        out.push('\n');
    }

    out
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

fn output_target_path_display(root: &Path, output_path: &Path) -> String {
    output_path
        .strip_prefix(root)
        .unwrap_or(output_path)
        .to_string_lossy()
        .replace('\\', "/")
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
    fn map_default_path_to_flat_file() {
        let rel = PathBuf::from("doc/software_development/first_experience/index.html");
        let target = map_html_rel_to_output(&rel).expect("mapping should succeed");
        assert_eq!(target.bucket, "default");
        assert_eq!(
            target.file_name,
            "software_development__first_experience.md"
        );
    }

    #[test]
    fn map_versioned_path_to_bucket() {
        let rel = PathBuf::from("doc/5.2.1/software_development/index.html");
        let target = map_html_rel_to_output(&rel).expect("mapping should succeed");
        assert_eq!(target.bucket, "5.2.1");
        assert_eq!(target.file_name, "software_development.md");
    }

    #[test]
    fn map_doc_root_to_index_md() {
        let rel = PathBuf::from("doc/index.html");
        let target = map_html_rel_to_output(&rel).expect("mapping should succeed");
        assert_eq!(target.bucket, "default");
        assert_eq!(target.file_name, "index.md");
    }

    #[test]
    fn map_version_root_to_index_md() {
        let rel = PathBuf::from("doc/5.2.1/index.html");
        let target = map_html_rel_to_output(&rel).expect("mapping should succeed");
        assert_eq!(target.bucket, "5.2.1");
        assert_eq!(target.file_name, "index.md");
    }

    #[test]
    fn extract_article_html_prefers_next_data_content() {
        let raw = r#"
        <html>
          <body>
            <script id="__NEXT_DATA__" type="application/json">
              {"props":{"pageProps":{"content":"<h1>Hello</h1><p>World</p>"}}}
            </script>
            <article><div class="prose"><h1>Fallback</h1></div></article>
          </body>
        </html>
        "#;
        let article = extract_article_html(raw);
        assert!(article.contains("<h1>Hello</h1>"));
        assert!(article.contains("<p>World</p>"));
    }

    #[test]
    fn keeps_hidden_heading_anchor() {
        let html =
            r#"<h2><span id="setup" style="display:block;position: absolute;"></span>Setup</h2>"#;
        let md = convert_article_html_to_markdown(html);
        assert!(md.contains("## Setup {#setup}"));
    }

    #[test]
    fn keeps_fenced_code_language() {
        let html = r#"<pre><code class="language-rust">fn main() {}</code></pre>"#;
        let md = convert_article_html_to_markdown(html);
        assert!(md.contains("```rust"));
        assert!(md.contains("fn main() {}"));
    }

    #[test]
    fn renders_definition_list() {
        let html = r#"<dl><dt id="t">Term</dt><dd>Definition A</dd><dd>Definition B</dd></dl>"#;
        let md = convert_article_html_to_markdown(html);
        assert!(md.contains("Term {#t}"));
        assert!(md.contains(": Definition A"));
        assert!(md.contains(": Definition B"));
    }

    #[test]
    fn renders_markdown_table() {
        let html = r#"<table><tr><th align="left">A</th><th align="right">B</th></tr><tr><td>x</td><td>y</td></tr></table>"#;
        let md = convert_article_html_to_markdown(html);
        assert!(md.contains("| A | B |"));
        assert!(md.contains("| :-- | --: |"));
        assert!(md.contains("| x | y |"));
    }

    #[test]
    fn sanitize_flat_segment_escapes_double_underscore() {
        assert_eq!(sanitize_flat_segment("foo__bar"), "foo____bar");
    }
}
