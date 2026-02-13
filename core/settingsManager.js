const storageKey = "ai_news_settings";
const apiBase = "";
let cachedSettings = null;

const defaultSettings = {
  general: {
    defaultModel: "auto",
    fallbackModel: "tfidf-pgn",
    summaryLength: "medium",
    outputFormat: "paragraph",
    languageStyle: "neutral"
  },
  analysis: {
    keyPhrases: true,
    namedEntities: true,
    sentiment: true,
    readability: true,
    questionAnswering: true
  },
  performance: {
    maxInputLength: 12000,
    chunkSize: 2000,
    streaming: false
  },
  extension: {
    compactMode: true,
    shortSummaryOnly: true,
    deepLink: true
  }
};

const modelOptions = [
  { id: "auto", label: "Auto (Smart Routing)" },
  { id: "tfidf-pgn", label: "TF-IDF + PGN (Existing)" },
  { id: "t5-small", label: "T5-Small (Pretrained)" },
  { id: "distilbart", label: "DistilBART (Pretrained)" },
  { id: "bart-large", label: "BART-Large (Pretrained)" },
  { id: "pegasus", label: "PEGASUS (Pretrained)" },
  { id: "t5-3b", label: "T5-3B (Pretrained)" }
];

const clone = (value) => JSON.parse(JSON.stringify(value));

const deepMerge = (base, next) => {
  const output = Array.isArray(base) ? base.slice() : { ...base };
  if (!next || typeof next !== "object") {
    return output;
  }
  Object.keys(next).forEach((key) => {
    if (next[key] && typeof next[key] === "object" && !Array.isArray(next[key])) {
      output[key] = deepMerge(base[key] || {}, next[key]);
    } else {
      output[key] = next[key];
    }
  });
  return output;
};

const loadRawSettings = () => {
  const raw = localStorage.getItem(storageKey);
  if (!raw) {
    return clone(defaultSettings);
  }
  try {
    return deepMerge(defaultSettings, JSON.parse(raw));
  } catch (error) {
    return clone(defaultSettings);
  }
};

const persistSettings = async (settings) => {
  cachedSettings = settings;
  localStorage.setItem(storageKey, JSON.stringify(settings));
  try {
    const response = await fetch(`${apiBase}/api/settings`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ settings })
    });
    if (response.ok) {
      const data = await response.json();
      const merged = deepMerge(defaultSettings, data.settings || data);
      cachedSettings = merged;
      localStorage.setItem(storageKey, JSON.stringify(merged));
    }
  } catch (error) {
  }
  window.dispatchEvent(new CustomEvent("settings:updated", { detail: cachedSettings }));
};

const fetchSettings = async () => {
  try {
    const response = await fetch(`${apiBase}/api/settings`);
    if (response.ok) {
      const data = await response.json();
      const merged = deepMerge(defaultSettings, data.settings || data);
      cachedSettings = merged;
      localStorage.setItem(storageKey, JSON.stringify(merged));
      return merged;
    }
  } catch (error) {
  }
  const fallback = loadRawSettings();
  cachedSettings = fallback;
  return fallback;
};

export const getSettings = async () => {
  if (cachedSettings) {
    return cachedSettings;
  }
  return fetchSettings();
};

export const saveSettings = async (partial) => {
  const current = await getSettings();
  const updated = deepMerge(current, partial);
  await persistSettings(updated);
  return cachedSettings;
};

export const resetSettings = async () => {
  await persistSettings(clone(defaultSettings));
  return cachedSettings;
};

export const onSettingsChange = (handler) => {
  window.addEventListener("settings:updated", (event) => handler(event.detail));
};

export const getModelOptions = () => modelOptions.slice();

export const getDefaultSettings = () => clone(defaultSettings);
