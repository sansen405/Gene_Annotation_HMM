export const API_BASE = import.meta.env.DEV ? "" : "http://localhost:5174";

export async function readJsonResponse(response) {
  const text = await response.text();
  if (!text) return {};
  try {
    return JSON.parse(text);
  } catch {
    if (text.includes("<!DOCTYPE") || text.includes("<html")) {
      throw connectionError();
    }
    return { error: text };
  }
}

function connectionError() {
  return new Error(
    "Cannot connect to Annotator on this machine. Restart the app to try again."
  );
}

export async function apiFetch(path, options = {}) {
  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, options);
  } catch {
    throw connectionError();
  }
  const payload = await readJsonResponse(response);
  if (!response.ok) {
    throw new Error(payload.detail || payload.error || `Request failed (${response.status}).`);
  }
  return payload;
}
