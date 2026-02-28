use super::{
    RagBundleArgs, RagBundleCommand, RagBundleCreateArgs, RagBundleExtractArgs,
    RagBundleInspectArgs,
};
use anyhow::{Context, Result, anyhow};
use chrono::{SecondsFormat, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha1::{Digest, Sha1};
use std::{
    fs::{self, File},
    io::Read,
    path::{Component, Path, PathBuf},
};

const CHUNKS_ROOT_DIRNAME: &str = "chunks";
const VECTORS_ROOT_DIRNAME: &str = "vectors_data";
const SCHEMA_VERSION: &str = "1";

macro_rules! vprintln {
    ($verbose:expr, $($arg:tt)*) => {
        if $verbose {
            println!($($arg)*);
        }
    };
}

#[derive(Debug, Clone, Copy)]
enum BundleType {
    Chunks,
    Vectors,
    Dual,
}

impl BundleType {
    fn as_str(self) -> &'static str {
        match self {
            BundleType::Chunks => "chunks",
            BundleType::Vectors => "vectors",
            BundleType::Dual => "dual",
        }
    }

    fn include_chunks(self) -> bool {
        matches!(self, BundleType::Chunks | BundleType::Dual)
    }

    fn include_vectors(self) -> bool {
        matches!(self, BundleType::Vectors | BundleType::Dual)
    }
}

impl std::str::FromStr for BundleType {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "chunks" => Ok(BundleType::Chunks),
            "vectors" => Ok(BundleType::Vectors),
            "dual" => Ok(BundleType::Dual),
            other => Err(anyhow!(
                "unsupported bundle type '{}', expected one of: chunks, vectors, dual",
                other
            )),
        }
    }
}

#[derive(Debug)]
struct SourceFile {
    source_path: PathBuf,
    archive_path: PathBuf,
    size_bytes: u64,
    sha1: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct BundleManifest {
    schema_version: String,
    bundle_type: String,
    created_at: String,
    chunks_manifest: Option<Value>,
    vectors_manifest: Option<Value>,
    files_total: usize,
    bytes_total: u64,
    files: Vec<BundleFileRecord>,
}

#[derive(Debug, Serialize, Deserialize)]
struct BundleFileRecord {
    path: String,
    size_bytes: u64,
    sha1: String,
}

pub fn run(args: RagBundleArgs) -> Result<()> {
    match args.command {
        RagBundleCommand::Create(create) => run_create(create),
        RagBundleCommand::Inspect(inspect) => run_inspect(inspect),
        RagBundleCommand::Extract(extract) => run_extract(extract),
    }
}

fn run_create(args: RagBundleCreateArgs) -> Result<()> {
    let verbose = args.verbose;
    let bundle_type = args.bundle_type.parse::<BundleType>()?;

    let home_dir =
        dirs::home_dir().ok_or_else(|| anyhow!("failed to resolve home directory for bundle"))?;
    let chunks_root = args
        .input_chunks
        .unwrap_or_else(|| home_dir.join(".aurora-grimoire").join(CHUNKS_ROOT_DIRNAME));
    let vectors_root = args
        .input_vectors
        .unwrap_or_else(|| home_dir.join(".aurora-grimoire").join(VECTORS_ROOT_DIRNAME));

    if bundle_type.include_chunks() && !chunks_root.exists() {
        return Err(anyhow!(
            "chunks directory does not exist: {}",
            chunks_root.display()
        ));
    }
    if bundle_type.include_vectors() && !vectors_root.exists() {
        return Err(anyhow!(
            "vectors directory does not exist: {}",
            vectors_root.display()
        ));
    }

    let mut source_files = Vec::new();
    if bundle_type.include_chunks() {
        collect_source_files(
            &chunks_root,
            Path::new("chunks"),
            &mut source_files,
            verbose,
        )?;
    }
    if bundle_type.include_vectors() {
        collect_source_files(
            &vectors_root,
            Path::new("vectors"),
            &mut source_files,
            verbose,
        )?;
    }
    source_files.sort_by(|a, b| a.archive_path.cmp(&b.archive_path));

    let chunks_manifest = if bundle_type.include_chunks() {
        read_optional_json(chunks_root.join("manifest.json"))?
    } else {
        None
    };
    let vectors_manifest = if bundle_type.include_vectors() {
        read_optional_json(vectors_root.join("manifest.json"))?
    } else {
        None
    };

    let files = source_files
        .iter()
        .map(|file| BundleFileRecord {
            path: file.archive_path.to_string_lossy().to_string(),
            size_bytes: file.size_bytes,
            sha1: file.sha1.clone(),
        })
        .collect::<Vec<_>>();
    let bytes_total = source_files.iter().map(|file| file.size_bytes).sum::<u64>();
    let manifest = BundleManifest {
        schema_version: SCHEMA_VERSION.to_string(),
        bundle_type: bundle_type.as_str().to_string(),
        created_at: now_rfc3339(),
        chunks_manifest,
        vectors_manifest,
        files_total: files.len(),
        bytes_total,
        files,
    };

    if let Some(parent) = args.out.parent() {
        fs::create_dir_all(parent).with_context(|| {
            format!(
                "failed to create output bundle parent directory {}",
                parent.display()
            )
        })?;
    }
    let output = File::create(&args.out)
        .with_context(|| format!("failed to create bundle file {}", args.out.display()))?;

    let encoder = zstd::Encoder::new(output, 3).context("failed to initialize zstd encoder")?;
    let mut tar = tar::Builder::new(encoder.auto_finish());

    for source in &source_files {
        tar.append_path_with_name(&source.source_path, &source.archive_path)
            .with_context(|| {
                format!(
                    "failed to append {} as {}",
                    source.source_path.display(),
                    source.archive_path.display()
                )
            })?;
    }

    let mut manifest_bytes =
        serde_json::to_vec_pretty(&manifest).context("failed to serialize bundle manifest")?;
    manifest_bytes.push(b'\n');
    let mut header = tar::Header::new_gnu();
    header.set_mode(0o644);
    header.set_size(manifest_bytes.len() as u64);
    header.set_cksum();
    tar.append_data(
        &mut header,
        "bundle_manifest.json",
        manifest_bytes.as_slice(),
    )
    .context("failed to append bundle manifest to archive")?;
    tar.finish().context("failed to finish writing bundle")?;

    let size_bytes = fs::metadata(&args.out)
        .with_context(|| format!("failed to stat bundle {}", args.out.display()))?
        .len();
    println!(
        "Bundle created: {} (type={}, size={})",
        args.out.display(),
        bundle_type.as_str(),
        size_bytes
    );
    Ok(())
}

fn run_inspect(args: RagBundleInspectArgs) -> Result<()> {
    let manifest = read_bundle_manifest(&args.file)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&manifest).context("failed to format bundle manifest")?
    );
    Ok(())
}

fn run_extract(args: RagBundleExtractArgs) -> Result<()> {
    let verbose = args.verbose;
    let home_dir =
        dirs::home_dir().ok_or_else(|| anyhow!("failed to resolve home directory for extract"))?;
    let output_root = args
        .out
        .unwrap_or_else(|| home_dir.join(".aurora-grimoire"));
    fs::create_dir_all(&output_root).with_context(|| {
        format!(
            "failed to create extraction root directory {}",
            output_root.display()
        )
    })?;

    let file = File::open(&args.file)
        .with_context(|| format!("failed to open bundle file {}", args.file.display()))?;
    let decoder = zstd::Decoder::new(file).with_context(|| {
        format!(
            "failed to initialize zstd decoder for {}",
            args.file.display()
        )
    })?;
    let mut archive = tar::Archive::new(decoder);
    let mut extracted = 0usize;

    for entry_result in archive.entries().context("failed to read bundle entries")? {
        let mut entry = entry_result.context("failed to read bundle archive entry")?;
        let rel_path = entry
            .path()
            .context("failed to read bundle entry path")?
            .into_owned();
        if !is_safe_relative_path(&rel_path) {
            return Err(anyhow!(
                "unsafe path in bundle entry: {}",
                rel_path.display()
            ));
        }

        let Some(target_path) = map_bundle_entry_to_extract_target(&output_root, &rel_path) else {
            continue;
        };
        if let Some(parent) = target_path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!(
                    "failed to create extraction directory {}",
                    parent.to_string_lossy()
                )
            })?;
        }
        entry.unpack(&target_path).with_context(|| {
            format!(
                "failed to extract {} to {}",
                rel_path.display(),
                target_path.display()
            )
        })?;
        extracted += 1;
        vprintln!(
            verbose,
            "[bundle] extracted {}",
            target_path.to_string_lossy()
        );
    }

    if extracted == 0 {
        return Err(anyhow!(
            "bundle {} did not contain extractable entries",
            args.file.display()
        ));
    }

    println!("Bundle extracted to {}", output_root.display());
    Ok(())
}

fn collect_source_files(
    root: &Path,
    archive_prefix: &Path,
    out: &mut Vec<SourceFile>,
    verbose: bool,
) -> Result<()> {
    if !root.is_dir() {
        return Err(anyhow!("input path is not a directory: {}", root.display()));
    }
    collect_source_files_recursive(root, root, archive_prefix, out, verbose)
}

fn collect_source_files_recursive(
    root: &Path,
    current: &Path,
    archive_prefix: &Path,
    out: &mut Vec<SourceFile>,
    verbose: bool,
) -> Result<()> {
    let entries = fs::read_dir(current)
        .with_context(|| format!("failed to read directory {}", current.display()))?;
    for entry in entries {
        let entry = entry
            .with_context(|| format!("failed to read directory entry in {}", current.display()))?;
        let path = entry.path();
        if path.is_dir() {
            collect_source_files_recursive(root, &path, archive_prefix, out, verbose)?;
            continue;
        }
        if !path.is_file() {
            continue;
        }
        let relative = path.strip_prefix(root).with_context(|| {
            format!(
                "failed to strip '{}' prefix from '{}'",
                root.display(),
                path.display()
            )
        })?;
        let archive_path = archive_prefix.join(relative);
        let (size_bytes, sha1) = file_stats_and_sha1(&path)?;
        out.push(SourceFile {
            source_path: path.clone(),
            archive_path,
            size_bytes,
            sha1,
        });
        vprintln!(verbose, "[bundle] add {}", path.to_string_lossy());
    }
    Ok(())
}

fn file_stats_and_sha1(path: &Path) -> Result<(u64, String)> {
    let metadata =
        fs::metadata(path).with_context(|| format!("failed to stat file {}", path.display()))?;
    let mut file =
        File::open(path).with_context(|| format!("failed to open file {}", path.display()))?;
    let mut hasher = Sha1::new();
    let mut buffer = [0u8; 8192];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("failed to read file {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok((metadata.len(), hex_lower(&hasher.finalize())))
}

fn read_optional_json(path: PathBuf) -> Result<Option<Value>> {
    if !path.exists() {
        return Ok(None);
    }
    let file = File::open(&path)
        .with_context(|| format!("failed to open JSON file {}", path.display()))?;
    let value: Value = serde_json::from_reader(file)
        .with_context(|| format!("failed to parse JSON file {}", path.display()))?;
    Ok(Some(value))
}

fn read_bundle_manifest(bundle_file: &Path) -> Result<BundleManifest> {
    let file = File::open(bundle_file)
        .with_context(|| format!("failed to open bundle file {}", bundle_file.display()))?;
    let decoder = zstd::Decoder::new(file)
        .with_context(|| format!("failed to decode bundle {}", bundle_file.display()))?;
    let mut archive = tar::Archive::new(decoder);
    for entry_result in archive.entries().context("failed to read bundle entries")? {
        let mut entry = entry_result.context("failed to read bundle archive entry")?;
        let path = entry
            .path()
            .context("failed to read bundle entry path")?
            .into_owned();
        if path == Path::new("bundle_manifest.json") {
            let mut buf = Vec::new();
            entry
                .read_to_end(&mut buf)
                .context("failed to read bundle_manifest.json content")?;
            let manifest: BundleManifest =
                serde_json::from_slice(&buf).context("failed to parse bundle manifest JSON")?;
            return Ok(manifest);
        }
    }
    Err(anyhow!(
        "bundle {} does not contain bundle_manifest.json",
        bundle_file.display()
    ))
}

fn map_bundle_entry_to_extract_target(root: &Path, rel_path: &Path) -> Option<PathBuf> {
    let mut components = rel_path.components();
    let first = components.next()?;
    match first {
        Component::Normal(name) if name == "chunks" => {
            let rest: PathBuf = components.collect();
            Some(root.join(CHUNKS_ROOT_DIRNAME).join(rest))
        }
        Component::Normal(name) if name == "vectors" => {
            let rest: PathBuf = components.collect();
            Some(root.join(VECTORS_ROOT_DIRNAME).join(rest))
        }
        _ => None,
    }
}

fn is_safe_relative_path(path: &Path) -> bool {
    if path.is_absolute() {
        return false;
    }
    !path
        .components()
        .any(|component| matches!(component, Component::ParentDir))
}

fn now_rfc3339() -> String {
    Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true)
}

fn hex_lower(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
