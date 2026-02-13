import { getSettings, onSettingsChange } from "/core/settingsManager.js";
import { summarizeText } from "/core/summarizer.js";

const form = document.querySelector(".form");
const urlInput = document.getElementById("url");
const fileUpload = document.getElementById("fileUpload");
const urlField = document.getElementById("urlField");
const fileField = document.getElementById("fileField");
const inputToggles = document.querySelectorAll(".input-toggle .toggle-pill");
const statusEl = document.getElementById("status");
const resultSection = document.getElementById("resultSection");
const resultTitle = document.getElementById("resultTitle");
const resultAuthors = document.getElementById("resultAuthors");
const resultDate = document.getElementById("resultDate");
const resultImage = document.getElementById("resultImage");
const resultSummary = document.getElementById("resultSummary");
const resultSentiment = document.getElementById("resultSentiment");
const resultKeyPoints = document.getElementById("resultKeyPoints");
const resultCard = document.getElementById("resultCard");
const resultToggle = document.getElementById("resultToggle");
let currentSettings = null;
let inputMode = "url";

const loadSettings = async () => {
  currentSettings = await getSettings();
  return currentSettings;
};

const setStatus = (text) => {
  if (statusEl) {
    statusEl.textContent = text;
  }
};

const setInputMode = (mode) => {
  inputMode = mode;
  inputToggles.forEach((toggle) => {
    toggle.classList.toggle("active", toggle.dataset.mode === mode);
  });
  if (urlField) {
    urlField.classList.toggle("is-hidden", mode !== "url");
  }
  if (fileField) {
    fileField.classList.toggle("is-hidden", mode !== "file");
  }
  if (mode === "url") {
    if (fileUpload) {
      fileUpload.value = "";
    }
  } else if (urlInput) {
    urlInput.value = "";
  }
};

const setVisible = (element, isVisible) => {
  if (!element) {
    return;
  }
  element.classList.toggle("is-hidden", !isVisible);
};

const clearChildren = (element) => {
  if (!element) {
    return;
  }
  while (element.firstChild) {
    element.removeChild(element.firstChild);
  }
};

const renderAuthors = (authorsText) => {
  clearChildren(resultAuthors);
  if (!authorsText) {
    return;
  }
  authorsText.split(",").map((author) => author.trim()).filter(Boolean).forEach((author, index) => {
    const li = document.createElement("li");
    li.textContent = author;
    li.classList.add("motion-item");
    li.style.setProperty("--delay", `${120 + index * 60}ms`);
    resultAuthors.appendChild(li);
  });
};

const renderKeyPoints = (summaryText) => {
  clearChildren(resultKeyPoints);
  if (!resultKeyPoints || !summaryText) {
    return;
  }
  const rawPoints = summaryText.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  let points = rawPoints;
  if (points.length <= 1) {
    points = summaryText.split(/[.!?]\s+/).map((line) => line.trim()).filter(Boolean);
  }
  points.slice(0, 4).forEach((point, index) => {
    const li = document.createElement("li");
    li.textContent = point;
    li.style.setProperty("--delay", `${160 + index * 70}ms`);
    resultKeyPoints.appendChild(li);
  });
};

const setResultExpanded = (isExpanded) => {
  if (resultCard) {
    resultCard.classList.toggle("expanded", isExpanded);
  }
  if (resultToggle) {
    resultToggle.setAttribute("aria-expanded", String(isExpanded));
    resultToggle.textContent = isExpanded ? "Hide details" : "Show details";
  }
};

const updateResult = (data) => {
  if (resultTitle) {
    resultTitle.textContent = data.title || "Summary";
  }
  renderAuthors(data.authors || "");
  if (resultDate) {
    resultDate.textContent = data.publish_date || "N/A";
  }
  if (resultImage) {
    if (data.top_image) {
      resultImage.src = data.top_image;
      resultImage.classList.remove("is-hidden");
    } else {
      resultImage.classList.add("is-hidden");
    }
  }
  if (resultSummary) {
    resultSummary.textContent = data.formatted_summary || data.summary || "";
  }
  renderKeyPoints(data.formatted_summary || data.summary || "");
  if (resultSentiment) {
    resultSentiment.textContent = data.sentiment || "N/A";
  }
  setVisible(resultSection, true);
  setResultExpanded(true);
  if (typeof window.toggleAuthors === "function") {
    window.toggleAuthors();
  }
  setStatus("");
};

const parseResponse = async (response) => {
  const text = await response.text();
  if (!text) {
    return { ok: response.ok, data: null, raw: "" };
  }
  try {
    return { ok: response.ok, data: JSON.parse(text), raw: text };
  } catch (error) {
    return { ok: response.ok, data: null, raw: text };
  }
};

const summarizeUrl = async (url) => {
  setStatus("Summarizing...");
  if (!currentSettings) {
    await loadSettings();
  }
  const data = await summarizeText({
    url,
    settings: currentSettings
  });
  updateResult(data);
};

const summarizeFile = async (file) => {
  setStatus("Summarizing...");
  if (!currentSettings) {
    await loadSettings();
  }
  const formData = new FormData();
  formData.append("file", file);
  formData.append("settings", JSON.stringify(currentSettings));
  const response = await fetch("/api/summarize-upload", {
    method: "POST",
    body: formData
  });
  const result = await parseResponse(response);
  if (!result.ok) {
    const message = result.data?.error || result.data?.message || result.raw || "Summarization failed";
    throw new Error(message);
  }
  if (!result.data) {
    throw new Error("Invalid response from server");
  }
  updateResult(result.data);
};

if (form) {
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const url = urlInput?.value?.trim();
    try {
      if (!currentSettings) {
        await loadSettings();
      }
      if (inputMode === "file") {
        const file = fileUpload?.files?.[0];
        if (!file) {
          setStatus("Upload a file to summarize");
          return;
        }
        await summarizeFile(file);
        return;
      }
      if (!url) {
        setStatus("Enter a valid URL to summarize");
        return;
      }
      await summarizeUrl(url);
    } catch (error) {
      setStatus(error.message || "Failed to summarize");
    }
  });
}

const start = async () => {
  await loadSettings();
  const params = new URLSearchParams(window.location.search);
  const paramUrl = params.get("url");
  if (paramUrl && urlInput) {
    urlInput.value = paramUrl;
    setInputMode("url");
    summarizeUrl(paramUrl).catch(() => {
      setStatus("Failed to summarize");
    });
  }
};

start();

onSettingsChange((settings) => {
  currentSettings = settings;
});

setInputMode("url");
inputToggles.forEach((toggle) => {
  toggle.addEventListener("click", () => {
    setInputMode(toggle.dataset.mode);
  });
});

if (resultToggle) {
  resultToggle.addEventListener("click", () => {
    const isExpanded = resultCard?.classList.contains("expanded");
    setResultExpanded(!isExpanded);
  });
}
