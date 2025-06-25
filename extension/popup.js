chrome.tabs.query({ active: true, currentWindow: true }, async function (tabs) {
  const url = tabs[0].url;
  
  const res = await fetch("https://your-render-api.onrender.com/check", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url: url })
  });

  const data = await res.json();
  document.getElementById("result").innerText =
    `🔍 ${data.result} News\nConfidence: ${(data.confidence * 100).toFixed(2)}%`;
});