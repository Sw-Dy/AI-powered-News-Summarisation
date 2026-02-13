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
  const payloadSettings = {
    ...settings,
    selectedModel: selectModel((text || "").length, settings || {})
  };
  const response = await fetch(`${apiBase || ""}/api/summarize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text,
      url,
      title,
      settings: {
        summaryLength: payloadSettings.summaryLength,
        maxInputLength: payloadSettings.maxInputLength,
        sentiment: payloadSettings.sentiment,
        selectedModel: payloadSettings.selectedModel
      }
    })
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Summarization failed");
  }
  return {
    ...data,
    selectedModel: payloadSettings.selectedModel,
    summary: formatSummary(data.summary, payloadSettings)
  };
};
