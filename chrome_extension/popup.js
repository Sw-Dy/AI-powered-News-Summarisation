import { getSettings, getModelOptions } from "./core/settingsManager.js";
import { summarizeText } from "./core/summarizer.js";

const summarizeButton = document.getElementById("summarize");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");
const modelSelect = document.getElementById("modelSelect");
const deepLink = document.getElementById("deepLink");

let currentSettings = getSettings();

const setStatus = (text) => {
  statusEl.textContent = text;
};

const setResult = (text) => {
  resultEl.textContent = text;
};

const extractPageData = async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id) {
    throw new Error("No active tab found");
  }
  const [{ result }] = await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: () => {
      const title = document.title || "";
      const url = location.href || "";
      const text = document.body ? document.body.innerText : "";
      return { title, url, text };
    }
  });
  return result;
};

const populateModels = () => {
  modelSelect.innerHTML = "";
  getModelOptions().forEach((model) => {
    const option = document.createElement("option");
    option.value = model.id;
    option.textContent = model.label;
    option.selected = model.id === currentSettings.general.defaultModel;
    modelSelect.appendChild(option);
  });
  if (currentSettings.extension.deepLink) {
    deepLink.classList.remove("is-hidden");
  } else {
    deepLink.classList.add("is-hidden");
  }
};

const summarize = async () => {
  summarizeButton.disabled = true;
  setStatus("Collecting page text...");
  setResult("");
  try {
    const pageData = await extractPageData();
    const trimmedText = (pageData.text || "").replace(/\s+/g, " ").trim();
    if (!trimmedText) {
      throw new Error("No readable text found on this page.");
    }
    const settings = {
      ...currentSettings.general,
      ...currentSettings.analysis,
      ...currentSettings.performance
    };
    settings.defaultModel = modelSelect.value;
    if (currentSettings.extension.shortSummaryOnly) {
      settings.summaryLength = "short";
    }
    const data = await summarizeText({
      text: trimmedText.slice(0, settings.maxInputLength || 12000),
      title: pageData.title || "",
      url: pageData.url || "",
      settings,
      apiBase: "http://127.0.0.1:5000"
    });
    const titleLine = data.title ? `${data.title}\n\n` : "";
    setResult(`${titleLine}${data.summary}`);
    setStatus("Done");
    if (currentSettings.extension.deepLink && deepLink) {
      const link = `http://127.0.0.1:5000/?url=${encodeURIComponent(pageData.url || "")}`;
      deepLink.href = link;
      deepLink.classList.remove("is-hidden");
    }
  } catch (error) {
    setStatus("Failed");
    setResult(error.message || "Something went wrong.");
  } finally {
    summarizeButton.disabled = false;
  }
};

modelSelect.addEventListener("change", () => {
  currentSettings = getSettings();
  if (currentSettings.extension.deepLink) {
    deepLink.classList.remove("is-hidden");
  } else {
    deepLink.classList.add("is-hidden");
  }
});

summarizeButton.addEventListener("click", summarize);

populateModels();
