import { selectModel } from "./modelRouter.js";

const bulletify = (summary) => {
  const sentences = summary.split(". ").map((part) => part.trim()).filter(Boolean);
  return sentences.map((sentence) => `• ${sentence.replace(/\.$/, "")}`).join("\n");
};

const tldrify = (summary) => {
  const firstSentence = summary.split(". ").map((part) => part.trim()).filter(Boolean)[0] || summary;
  return `TL;DR: ${firstSentence.replace(/\.$/, "")}.`;
};

export const formatSummary = (summary, settings) => {
  const outputFormat = settings?.outputFormat || "paragraph";
  if (outputFormat === "bullets") {
    return bulletify(summary);
  }
  if (outputFormat === "tldr") {
    return tldrify(summary);
  }
  return summary;
};

export const summarizeText = async ({ text, url, title, settings, apiBase }) => {
  const baseSettings = settings || {};
  const flatSettings = {
    ...(baseSettings.general || {}),
    ...(baseSettings.analysis || {}),
    ...(baseSettings.performance || {})
  };
  const selectedModel = selectModel((text || "").length, flatSettings);
  const payloadSettings = {
    ...baseSettings,
    selectedModel
  };
  const response = await fetch(`${apiBase || ""}/api/summarize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text,
      url,
      title,
      settings: payloadSettings
    })
  });
  const raw = await response.text();
  let data = null;
  if (raw) {
    try {
      data = JSON.parse(raw);
    } catch (error) {
      data = null;
    }
  }
  if (!response.ok) {
    const message = data?.error || data?.message || raw || "Summarization failed";
    throw new Error(message);
  }
  if (!data) {
    throw new Error("Invalid response from server");
  }
  return {
    ...data,
    selectedModel,
    summary: formatSummary(data.summary, flatSettings)
  };
};
