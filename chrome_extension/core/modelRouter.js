export const selectModel = (articleLength, userPreferences = {}) => {
  const defaultModel = userPreferences.defaultModel || "auto";
  const fallbackModel = userPreferences.fallbackModel || "tfidf-pgn";
  if (defaultModel && defaultModel !== "auto") {
    return defaultModel;
  }
  if (articleLength <= 1200) {
    return "t5-small";
  }
  if (articleLength <= 4000) {
    return "distilbart";
  }
  if (articleLength <= 8000) {
    return "bart-large";
  }
  return fallbackModel || "pegasus";
};

export const applyFallback = (primary, fallback) => {
  if (primary) {
    return primary;
  }
  return fallback || "tfidf-pgn";
};
