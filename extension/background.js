chrome.runtime.onInstalled.addListener(() => {
    chrome.contextMenus.create({
        id: "checkClaim",
        title: "Kiểm tra claim bị bôi đen",
        contexts: ["selection"]
    });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
    if (info.menuItemId === "checkClaim") {
        const text = info.selectionText;
        if (text) {
            // First, send a message to content script to show a "Loading..." notification
            chrome.tabs.sendMessage(tab.id, { action: "showLoading" });

            // Call the local API
            fetch("http://localhost:8000/verify", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ claim: text })
            })
                .then(response => response.json())
                .then(data => {
                    // Send result to content script
                    chrome.tabs.sendMessage(tab.id, { action: "showResult", verdict: data.verdict });
                })
                .catch(error => {
                    console.error("Error:", error);
                    chrome.tabs.sendMessage(tab.id, { action: "showResult", verdict: "Lỗi kết nối tới server" });
                });
        }
    }
});
