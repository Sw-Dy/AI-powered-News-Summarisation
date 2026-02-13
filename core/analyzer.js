export const analyzeText = (text, settings = {}) => {
  const length = text ? text.length : 0;
  const wordCount = text ? text.split(/\s+/).filter(Boolean).length : 0;
  const result = {
    keyPhrases: settings.keyPhrases ? [] : null,
    namedEntities: settings.namedEntities ? [] : null,
    readability: settings.readability ? Math.max(0, Math.min(100, 100 - length / 120)) : null,
    sentiment: settings.sentiment ? null : null,
    questionAnswering: settings.questionAnswering ? [] : null,
    wordCount
  };
  return result;
};
