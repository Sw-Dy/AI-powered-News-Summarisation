import { getSettings, saveSettings, resetSettings, getModelOptions } from "/core/settingsManager.js";

const summaryLength = document.getElementById("summaryLength");
const outputFormat = document.getElementById("outputFormat");
const languageStyle = document.getElementById("languageStyle");
const maxInputLength = document.getElementById("maxInputLength");
const chunkSize = document.getElementById("chunkSize");
const keyPhrases = document.getElementById("keyPhrases");
const namedEntities = document.getElementById("namedEntities");
const sentiment = document.getElementById("sentiment");
const readability = document.getElementById("readability");
const questionAnswering = document.getElementById("questionAnswering");
const streaming = document.getElementById("streaming");
const compactMode = document.getElementById("compactMode");
const shortSummaryOnly = document.getElementById("shortSummaryOnly");
const deepLink = document.getElementById("deepLink");
const defaultModelCards = document.getElementById("defaultModelCards");
const fallbackModelCards = document.getElementById("fallbackModelCards");
const saveStatus = document.getElementById("saveStatus");

const setStatus = (text) => {
  if (saveStatus) {
    saveStatus.textContent = text;
  }
};

const applyToInputs = (settings) => {
  summaryLength.value = settings.general.summaryLength;
  outputFormat.value = settings.general.outputFormat;
  languageStyle.value = settings.general.languageStyle;
  maxInputLength.value = settings.performance.maxInputLength;
  chunkSize.value = settings.performance.chunkSize;
  keyPhrases.checked = settings.analysis.keyPhrases;
  namedEntities.checked = settings.analysis.namedEntities;
  sentiment.checked = settings.analysis.sentiment;
  readability.checked = settings.analysis.readability;
  questionAnswering.checked = settings.analysis.questionAnswering;
  streaming.checked = settings.performance.streaming;
  compactMode.checked = settings.extension.compactMode;
  shortSummaryOnly.checked = settings.extension.shortSummaryOnly;
  deepLink.checked = settings.extension.deepLink;
};

const renderModelCards = (container, groupName, selected) => {
  container.innerHTML = "";
  getModelOptions().forEach((model) => {
    const label = document.createElement("label");
    label.className = "model-card";
    const input = document.createElement("input");
    input.type = "radio";
    input.name = groupName;
    input.value = model.id;
    input.checked = model.id === selected;
    const span = document.createElement("span");
    span.textContent = model.label;
    label.appendChild(input);
    label.appendChild(span);
    if (input.checked) {
      label.classList.add("active");
    }
    input.addEventListener("change", async () => {
      container.querySelectorAll(".model-card").forEach((card) => card.classList.remove("active"));
      label.classList.add("active");
      if (groupName === "defaultModel") {
        await saveSettings({ general: { defaultModel: model.id } });
      } else {
        await saveSettings({ general: { fallbackModel: model.id } });
      }
      setStatus("Saved");
    });
    container.appendChild(label);
  });
};

const bindInputs = () => {
  summaryLength.addEventListener("change", async () => {
    await saveSettings({ general: { summaryLength: summaryLength.value } });
    setStatus("Saved");
  });
  outputFormat.addEventListener("change", async () => {
    await saveSettings({ general: { outputFormat: outputFormat.value } });
    setStatus("Saved");
  });
  languageStyle.addEventListener("change", async () => {
    await saveSettings({ general: { languageStyle: languageStyle.value } });
    setStatus("Saved");
  });
  maxInputLength.addEventListener("change", async () => {
    await saveSettings({ performance: { maxInputLength: Number(maxInputLength.value) } });
    setStatus("Saved");
  });
  chunkSize.addEventListener("change", async () => {
    await saveSettings({ performance: { chunkSize: Number(chunkSize.value) } });
    setStatus("Saved");
  });
  keyPhrases.addEventListener("change", async () => {
    await saveSettings({ analysis: { keyPhrases: keyPhrases.checked } });
    setStatus("Saved");
  });
  namedEntities.addEventListener("change", async () => {
    await saveSettings({ analysis: { namedEntities: namedEntities.checked } });
    setStatus("Saved");
  });
  sentiment.addEventListener("change", async () => {
    await saveSettings({ analysis: { sentiment: sentiment.checked } });
    setStatus("Saved");
  });
  readability.addEventListener("change", async () => {
    await saveSettings({ analysis: { readability: readability.checked } });
    setStatus("Saved");
  });
  questionAnswering.addEventListener("change", async () => {
    await saveSettings({ analysis: { questionAnswering: questionAnswering.checked } });
    setStatus("Saved");
  });
  streaming.addEventListener("change", async () => {
    await saveSettings({ performance: { streaming: streaming.checked } });
    setStatus("Saved");
  });
  compactMode.addEventListener("change", async () => {
    await saveSettings({ extension: { compactMode: compactMode.checked } });
    setStatus("Saved");
  });
  shortSummaryOnly.addEventListener("change", async () => {
    await saveSettings({ extension: { shortSummaryOnly: shortSummaryOnly.checked } });
    setStatus("Saved");
  });
  deepLink.addEventListener("change", async () => {
    await saveSettings({ extension: { deepLink: deepLink.checked } });
    setStatus("Saved");
  });
};

const initialize = async () => {
  const settings = await getSettings();
  applyToInputs(settings);
  renderModelCards(defaultModelCards, "defaultModel", settings.general.defaultModel);
  renderModelCards(fallbackModelCards, "fallbackModel", settings.general.fallbackModel);
  bindInputs();
};

document.getElementById("saveSettings").addEventListener("click", async () => {
  await saveSettings({});
  setStatus("Saved");
});

document.getElementById("resetSettings").addEventListener("click", async () => {
  const settings = await resetSettings();
  applyToInputs(settings);
  renderModelCards(defaultModelCards, "defaultModel", settings.general.defaultModel);
  renderModelCards(fallbackModelCards, "fallbackModel", settings.general.fallbackModel);
  setStatus("Reset to defaults");
});

initialize();
