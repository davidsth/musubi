use crate::control::AcceleratorInfo;

pub fn memory_label(gpu: &AcceleratorInfo) -> String {
    gpu.memory_bytes.map(|n| format!("{:.1} GiB", n as f64 / 1024.0_f64.powi(3)))
        .unwrap_or_else(|| if gpu.unified_memory { "shared unified memory".into() } else { "memory unknown".into() })
}

pub fn detect_accelerators() -> Vec<AcceleratorInfo> {
    if let Some(found) = nvidia_smi() { return found; }
    #[cfg(target_os = "macos")] { if let Some(found) = macos() { return found; } }
    #[cfg(target_os = "windows")] { if let Some(found) = windows() { return found; } }
    #[cfg(target_os = "linux")] { if let Some(found) = linux() { return found; } }
    Vec::new()
}

fn nvidia_smi() -> Option<Vec<AcceleratorInfo>> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]).output().ok()?;
    if !out.status.success() { return None; }
    let found: Vec<_> = String::from_utf8_lossy(&out.stdout).lines().filter_map(|line| {
        let (name, mem) = line.split_once(',')?;
        let name = name.trim();
        if name.is_empty() { return None; }
        Some(AcceleratorInfo { name: name.into(), vendor: "NVIDIA".into(),
            memory_bytes: mem.trim().parse::<u64>().ok().map(|v| v * 1024 * 1024), unified_memory: false })
    }).take(8).collect();
    if found.is_empty() { None } else { Some(found) }
}

#[cfg(target_os = "macos")]
fn macos() -> Option<Vec<AcceleratorInfo>> {
    let out = std::process::Command::new("system_profiler").arg("SPDisplaysDataType").output().ok()?;
    if !out.status.success() { return None; }
    let found: Vec<_> = String::from_utf8_lossy(&out.stdout).lines().filter_map(|line| {
        let (key, value) = line.trim().split_once(':')?;
        if key != "Chipset Model" || value.trim().is_empty() { return None; }
        let name = value.trim().to_string();
        Some(AcceleratorInfo { vendor: if name.starts_with("Apple") { "Apple".into() } else { "Unknown".into() },
            unified_memory: name.starts_with("Apple"), name, memory_bytes: None })
    }).take(8).collect();
    if found.is_empty() { None } else { Some(found) }
}

#[cfg(target_os = "windows")]
fn windows() -> Option<Vec<AcceleratorInfo>> {
    let out = std::process::Command::new("powershell").args(["-NoProfile", "-Command",
        "Get-CimInstance Win32_VideoController | Select Name,AdapterRAM | ConvertTo-Json -Compress"]).output().ok()?;
    if !out.status.success() { return None; }
    let value: serde_json::Value = serde_json::from_slice(&out.stdout).ok()?;
    let rows = value.as_array().cloned().unwrap_or_else(|| vec![value]);
    let found: Vec<_> = rows.iter().filter_map(|v| {
        let name = v.get("Name")?.as_str()?.trim();
        if name.is_empty() { return None; }
        let lower = name.to_lowercase();
        let vendor = if lower.contains("nvidia") { "NVIDIA" } else if lower.contains("amd") || lower.contains("radeon") { "AMD" } else if lower.contains("intel") { "Intel" } else { "Unknown" };
        Some(AcceleratorInfo { name: name.into(), vendor: vendor.into(),
            memory_bytes: v.get("AdapterRAM").and_then(|x| x.as_u64()), unified_memory: false })
    }).take(8).collect();
    if found.is_empty() { None } else { Some(found) }
}

#[cfg(target_os = "linux")]
fn linux() -> Option<Vec<AcceleratorInfo>> {
    let out = std::process::Command::new("lspci").output().ok()?;
    if !out.status.success() { return None; }
    let found: Vec<_> = String::from_utf8_lossy(&out.stdout).lines().filter(|line| {
        let l = line.to_lowercase(); l.contains("vga compatible controller") || l.contains("3d controller") || l.contains("display controller")
    }).filter_map(|line| {
        let name = line.rsplit_once(": ")?.1.trim();
        let l = name.to_lowercase();
        let vendor = if l.contains("nvidia") { "NVIDIA" } else if l.contains("amd") || l.contains("radeon") { "AMD" } else if l.contains("intel") { "Intel" } else { "Unknown" };
        Some(AcceleratorInfo { name: name.into(), vendor: vendor.into(), memory_bytes: None, unified_memory: false })
    }).take(8).collect();
    if found.is_empty() { None } else { Some(found) }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unified_memory_is_not_misreported_as_vram() {
        let gpu = AcceleratorInfo { name: "Apple GPU".into(), vendor: "Apple".into(), memory_bytes: None, unified_memory: true };
        assert_eq!(memory_label(&gpu), "shared unified memory");
    }
}
