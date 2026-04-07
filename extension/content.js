chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "showLoading") {
        showNotification("Đang phân tích claim...", "loading");
    } else if (request.action === "showResult") {
        showNotification(`Kết quả: <b>${request.verdict}</b>`, "result");
    }
});

function showNotification(message, type) {
    // Remove existing notification if it exists
    let existingNotification = document.getElementById("crypto-claim-notification");
    if (existingNotification) {
        existingNotification.remove();
    }

    const notification = document.createElement("div");
    notification.id = "crypto-claim-notification";

    // Style the notification
    let bgColor = "#3b82f6"; // blue for loading
    let textColor = "white";
    if (type === "result") {
        if (message.includes("Đúng") || message.includes("True")) {
            bgColor = "#10b981"; // green
        } else if (message.includes("Sai") || message.includes("False")) {
            bgColor = "#ef4444"; // red
        } else {
            bgColor = "#f59e0b"; // yellow for error
        }
    }

    Object.assign(notification.style, {
        position: "fixed",
        top: "20px",
        right: "20px",
        padding: "16px 24px",
        backgroundColor: bgColor,
        color: textColor,
        borderRadius: "8px",
        boxShadow: "0 4px 6px rgba(0,0,0,0.1)",
        zIndex: "2147483647", // Max z-index
        fontFamily: "'Helvetica Neue', Helvetica, Arial, sans-serif",
        fontSize: "16px",
        minWidth: "250px",
        textAlign: "center",
        transition: "all 0.3s ease",
        opacity: "0",
        transform: "translateY(-20px)"
    });

    notification.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="flex-grow: 1;">${message}</span>
            <button id="crypto-claim-close" style="margin-left: 20px; background: none; border: none; color: white; cursor: pointer; font-size: 20px; font-weight: bold; line-height: 1;">&times;</button>
        </div>
    `;

    document.body.appendChild(notification);

    // Animate in
    setTimeout(() => {
        notification.style.opacity = "1";
        notification.style.transform = "translateY(0)";
    }, 10);

    // Close button logic
    document.getElementById("crypto-claim-close").addEventListener("click", () => {
        notification.style.opacity = "0";
        setTimeout(() => notification.remove(), 300);
    });

    // Auto-remove after some time only if it's a result
    if (type === "result") {
        setTimeout(() => {
            if (document.body.contains(notification)) {
                notification.style.opacity = "0";
                setTimeout(() => notification.remove(), 300);
            }
        }, 10000);
    }
}
