chrome.runtime.onInstalled.addListener(() => {
    chrome.contextMenus.create({
        id: "checkClaim",
        title: "Kiểm tra claim bị bôi đen",
        contexts: ["selection"]
    });
});

async function ensureContentScript(tabId) {
    return new Promise((resolve) => {
        chrome.tabs.sendMessage(tabId, { action: "ping" }, (response) => {
            if (chrome.runtime.lastError || !response) {
                chrome.scripting.executeScript(
                    { target: { tabId }, files: ["content.js"] },
                    () => resolve()
                );
            } else {
                resolve();
            }
        });
    });
}

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
    if (info.menuItemId !== "checkClaim") return;
    const text = info.selectionText;
    if (!text) return;

    await ensureContentScript(tab.id);

    // Chạy fetch trực tiếp trong tab (page context) thay vì service worker
    // → tránh bị Chrome terminate SW sau ~30s khi inference đang chạy
    chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: (claim) => {
            window.__claimShowLoading && window.__claimShowLoading();

            fetch("http://localhost:8000/verify", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ claim })
            })
                .then(r => r.json())
                .then(data => {
                    window.__claimShowResult && window.__claimShowResult(data);
                })
                .catch(() => {
                    window.__claimShowResult && window.__claimShowResult("Lỗi kết nối tới server");
                });
        },
        args: [text]
    });
});
