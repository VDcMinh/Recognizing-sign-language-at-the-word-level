export async function fetchSettings() {
  const response = await fetch("/api/settings");
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "Could not load backend settings.");
  }
  return payload;
}

export async function predictVideo({ file, branch }) {
  const formData = new FormData();
  formData.append("video", file);
  formData.append("branch", branch);

  const response = await fetch("/api/predict", {
    method: "POST",
    body: formData
  });
  const payload = await response.json();
  if (!response.ok || !payload.ok) {
    throw new Error(payload.error || "Prediction failed.");
  }
  return payload.result;
}
