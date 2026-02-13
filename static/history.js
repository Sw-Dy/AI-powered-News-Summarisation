const historyList = document.getElementById("historyList");
const historyDetail = document.getElementById("historyDetail");
const historyDetailTitle = document.getElementById("historyDetailTitle");
const historyDetailMeta = document.getElementById("historyDetailMeta");
const historyDetailUrl = document.getElementById("historyDetailUrl");
const historyDetailSummary = document.getElementById("historyDetailSummary");
const historyDetailSentiment = document.getElementById("historyDetailSentiment");
const historyDetailAuthors = document.getElementById("historyDetailAuthors");
const historyDetailDate = document.getElementById("historyDetailDate");
const historyDetailSettings = document.getElementById("historyDetailSettings");

const setVisible = (element, isVisible) => {
  if (!element) {
    return;
  }
  element.classList.toggle("is-hidden", !isVisible);
};

const setHistoryDetail = (item) => {
  if (!historyDetail) {
    return;
  }
  if (!item) {
    historyDetailTitle.textContent = "Select a summary";
    historyDetailMeta.textContent = "";
    historyDetailUrl.textContent = "";
    historyDetailUrl.removeAttribute("href");
    historyDetailSummary.textContent = "";
    historyDetailSentiment.textContent = "";
    historyDetailAuthors.textContent = "";
    historyDetailDate.textContent = "";
    historyDetailSettings.textContent = "";
    setVisible(historyDetail, true);
    return;
  }
  historyDetailTitle.textContent = item.title || item.url || "Summary";
  const metaParts = [];
  if (item.created_at) {
    metaParts.push(item.created_at);
  }
  if (item.selected_model) {
    metaParts.push(item.selected_model);
  }
  historyDetailMeta.textContent = metaParts.join(" • ");
  if (item.url) {
    historyDetailUrl.textContent = item.url;
    historyDetailUrl.href = item.url;
  } else {
    historyDetailUrl.textContent = "";
    historyDetailUrl.removeAttribute("href");
  }
  historyDetailSummary.textContent = item.summary || "";
  historyDetailSentiment.textContent = item.sentiment ? `Sentiment: ${item.sentiment}` : "";
  historyDetailAuthors.textContent = item.authors ? `Authors: ${item.authors}` : "";
  historyDetailDate.textContent = item.publish_date ? `Publication Date: ${item.publish_date}` : "";
  historyDetailSettings.textContent = item.settings ? JSON.stringify(item.settings, null, 2) : "";
  setVisible(historyDetail, true);
};

const renderHistory = (items) => {
  if (!historyList) {
    return;
  }
  historyList.innerHTML = "";
  if (!items || items.length === 0) {
    historyList.textContent = "No summaries yet.";
    setHistoryDetail(null);
    return;
  }
  items.forEach((item) => {
    const wrapper = document.createElement("button");
    wrapper.type = "button";
    wrapper.className = "history-item";
    wrapper.addEventListener("click", () => {
      setHistoryDetail(item);
    });
    const title = document.createElement("div");
    title.textContent = item.title || item.url || "Summary";
    const meta = document.createElement("div");
    const metaParts = [];
    if (item.created_at) {
      metaParts.push(item.created_at);
    }
    if (item.selected_model) {
      metaParts.push(item.selected_model);
    }
    meta.textContent = metaParts.join(" • ");
    const summary = document.createElement("div");
    summary.textContent = item.summary || "";
    wrapper.appendChild(title);
    wrapper.appendChild(meta);
    wrapper.appendChild(summary);
    historyList.appendChild(wrapper);
  });
  setHistoryDetail(items[0]);
};

const fetchHistory = async () => {
  try {
    const response = await fetch("/api/history?limit=50");
    if (!response.ok) {
      return;
    }
    const data = await response.json();
    renderHistory(data.items || []);
  } catch (error) {
  }
};

fetchHistory();
