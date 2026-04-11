chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "ping") {
        sendResponse({ status: "ok" });
    }
});

// Expose helpers cho background.js gọi qua executeScript
window.__claimShowLoading = () => showNotification({ status: "loading", message: "Đang phân tích dữ liệu và tìm kiếm nguồn..." });
window.__claimShowResult = (data) => {
    if (typeof data === "string") {
        // Fallback for errors
        showNotification({ status: "error", message: data });
    } else {
        showNotification(data);
    }
};

function showNotification(data) {
    let existingNotification = document.getElementById("crypto-claim-notification-wrapper");
    if (existingNotification) {
        existingNotification.remove();
    }

    const wrapper = document.createElement("div");
    wrapper.id = "crypto-claim-notification-wrapper";

    // Inject Fonts
    if (!document.getElementById("crypto-claim-fonts")) {
        const fontLink = document.createElement("link");
        fontLink.id = "crypto-claim-fonts";
        fontLink.href = "https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap";
        fontLink.rel = "stylesheet";
        document.head.appendChild(fontLink);
    }

    let theme = {
        glow: "rgba(66, 133, 244, 0.4)",
        iconColor: "#4285F4",
        gradientUrl: "linear-gradient(135deg, #4285F4 0%, #34A853 100%)",
        iconSvg: `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" class="crypto-spin"><circle cx="12" cy="12" r="10" stroke="#e0e0e0" stroke-width="3"></circle><path d="M12 2a10 10 0 0 1 10 10" stroke="#4285F4" stroke-width="3" stroke-linecap="round"></path></svg>`
    };

    let titleText = "Đang kiểm tra...";
    let contentHtml = `<div style="padding-top: 12px; font-size: 15px; color: #5f6368; line-height: 1.5; font-weight: 400;">${data.message || "Đang phân tích thông tin bằng AI..."}</div>`;

    if (data.status === "error") {
        theme.glow = "rgba(234, 67, 53, 0.4)";
        theme.iconColor = "#EA4335";
        theme.gradientUrl = "linear-gradient(135deg, #EA4335 0%, #FBBC04 100%)";
        theme.iconSvg = `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="${theme.iconColor}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>`;
        titleText = "Đã xảy ra lỗi";
        contentHtml = `<div style="padding-top: 12px; font-size: 15px; color: #5f6368; font-weight: 400;">${data.message}</div>`;
    } else if (data.verdict) {
        const isTrue = data.verdict === "Đúng" || data.verdict === "True";
        const isUncertain = data.verdict === "Chưa chắc chắn" || data.verdict === "Not enough evidence";

        if (isTrue) {
            theme.glow = "rgba(52, 168, 83, 0.45)";
            theme.iconColor = "#34A853";
            theme.gradientUrl = "linear-gradient(135deg, #34A853 0%, #0F9D58 100%)";
            theme.iconSvg = `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="${theme.iconColor}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>`;
        } else if (isUncertain) {
            theme.glow = "rgba(251, 188, 4, 0.45)";
            theme.iconColor = "#FBBC04";
            theme.gradientUrl = "linear-gradient(135deg, #FBBC04 0%, #F29900 100%)";
            theme.iconSvg = `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="${theme.iconColor}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>`;
        } else {
            theme.glow = "rgba(234, 67, 53, 0.45)";
            theme.iconColor = "#EA4335";
            theme.gradientUrl = "linear-gradient(135deg, #EA4335 0%, #C5221F 100%)";
            theme.iconSvg = `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="${theme.iconColor}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2"></polygon><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg>`;
        }

        titleText = "Kết quả phân tích";
        const confPercent = data.confidence ? (data.confidence * 100).toFixed(1) + "%" : "";

        // Tối ưu UI hiển thị Link
        let evidenceHtml = "";
        let sources = data.source_links || [];

        if (sources.length > 0) {
            evidenceHtml = `<div style="margin-top: 16px; position: relative;">`;
            evidenceHtml += `<div style="font-size: 11px; font-weight: 600; color: #80868b; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.8px;">Nguồn kiểm chứng</div>`;
            evidenceHtml += `<div class="crypto-scrollbar" style="max-height: 140px; overflow-y: auto; padding-right: 4px; display: flex; flex-direction: column; gap: 8px;">`;

            sources.forEach(url => {
                let displayUrl = url;
                try {
                    displayUrl = new URL(url).hostname.replace('www.', '');
                } catch (e) {
                    if (displayUrl.length > 40) displayUrl = displayUrl.substring(0, 40) + "...";
                }

                evidenceHtml += `
                <a href="${url}" target="_blank" class="crypto-link-item" style="display: flex; align-items: center; padding: 10px 14px; background: rgba(248, 249, 250, 0.6); backdrop-filter: blur(4px); border: 1px solid rgba(0,0,0,0.04); border-radius: 10px; text-decoration: none; transition: all 0.2s ease; gap: 10px;">
                    <div style="flex-shrink: 0; width: 24px; height: 24px; border-radius: 6px; background: ${theme.gradientUrl}; display: flex; align-items: center; justify-content: center;">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"></path><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"></path></svg>
                    </div>
                    <span style="color: #3c4043; font-size: 13px; font-weight: 500; font-family: 'Outfit', sans-serif; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 100%;">${displayUrl}</span>
                </a>`;
            });
            evidenceHtml += `</div></div>`;
        }

        contentHtml = `
            <div style="display: flex; align-items: stretch; margin-top: 12px; gap: 10px; padding: 12px; background: rgba(255,255,255,0.5); border-radius: 12px; border: 1px solid rgba(255,255,255,0.8);">
                <div style="flex: 1; display: flex; flex-direction: column; justify-content: center; align-items: center; background: ${theme.gradientUrl}; -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 24px; font-weight: 700; letter-spacing: -0.5px;">
                    ${data.verdict}
                </div>
                ${confPercent ? `
                <div style="width: 1px; background: rgba(0,0,0,0.06); margin: 4px 0;"></div>
                <div style="flex: 1; display: flex; flex-direction: column; justify-content: center; align-items: center;">
                    <span style="font-size: 11px; color: #80868b; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px;">Độ tin cậy</span>
                    <span style="font-size: 18px; color: #3c4043; font-weight: 600;">${confPercent}</span>
                </div>` : ''}
            </div>
            ${evidenceHtml}
        `;
    }

    // CSS Magic
    if (!document.getElementById("crypto-claim-styles")) {
        const style = document.createElement("style");
        style.id = "crypto-claim-styles";
        style.innerHTML = `
            @keyframes cryptoPulse {
                0% { box-shadow: 0 0 0 0 rgba(66, 133, 244, 0.4); }
                70% { box-shadow: 0 0 0 10px rgba(66, 133, 244, 0); }
                100% { box-shadow: 0 0 0 0 rgba(66, 133, 244, 0); }
            }
            @keyframes cryptoSpin {
                100% { transform: rotate(360deg); }
            }
            @keyframes cryptoSlideIn {
                0% { opacity: 0; transform: translateY(20px) scale(0.95); filter: blur(5px); }
                100% { opacity: 1; transform: translateY(0) scale(1); filter: blur(0); }
            }
            
            #crypto-claim-notification-wrapper {
                animation: cryptoSlideIn 0.5s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
            }
            .crypto-spin {
                animation: cryptoSpin 1s linear infinite;
            }
            .crypto-link-item:hover {
                background: white !important;
                border-color: rgba(66, 133, 244, 0.3) !important;
                box-shadow: 0 4px 12px rgba(0,0,0,0.05);
                transform: translateY(-1px);
            }
            .crypto-scrollbar::-webkit-scrollbar {
                width: 4px;
            }
            .crypto-scrollbar::-webkit-scrollbar-track {
                background: transparent; 
            }
            .crypto-scrollbar::-webkit-scrollbar-thumb {
                background: rgba(0,0,0,0.1); 
                border-radius: 10px;
            }
            .crypto-scrollbar::-webkit-scrollbar-thumb:hover {
                background: rgba(0,0,0,0.2); 
            }
            .crypto-close-btn {
                transition: all 0.2s ease;
                background: rgba(0,0,0,0.03);
            }
            .crypto-close-btn:hover {
                background: rgba(234, 67, 53, 0.1);
                color: #EA4335 !important;
                transform: scale(1.1);
            }
        `;
        document.head.appendChild(style);
    }

    // Glassmorphism Wrapper
    Object.assign(wrapper.style, {
        position: "fixed",
        top: "30px",
        right: "30px",
        width: "360px",
        background: "rgba(255, 255, 255, 0.85)",
        backdropFilter: "blur(20px)",
        WebkitBackdropFilter: "blur(20px)",
        borderRadius: "20px",
        boxShadow: `0 10px 30px -10px rgba(0, 0, 0, 0.15), 0 0 0 1px rgba(255, 255, 255, 0.6) inset, 0 0 20px 0 ${theme.glow}`,
        zIndex: "2147483647",
        fontFamily: "'Outfit', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
        color: "#202124",
        padding: "20px",
        overflow: "visible",
        opacity: "0" // For animation start state
    });

    wrapper.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="display: flex; justify-content: center; align-items: center; width: 36px; height: 36px; background: rgba(255,255,255,0.9); box-shadow: 0 4px 10px rgba(0,0,0,0.05); border-radius: 12px; ${data.status === 'loading' ? 'animation: cryptoPulse 2s infinite;' : ''}">
                    ${theme.iconSvg}
                </div>
                <span style="font-size: 16px; font-weight: 600; color: #202124;">${titleText}</span>
            </div>
            <button id="crypto-claim-close" class="crypto-close-btn" style="border: none; padding: 6px; border-radius: 50%; color: #80868b; cursor: pointer; display: flex; align-items: center; justify-content: center; outline: none;">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
            </button>
        </div>
        ${contentHtml}
    `;

    document.body.appendChild(wrapper);

    // Close button logic
    document.getElementById("crypto-claim-close").addEventListener("click", () => {
        wrapper.style.opacity = "0";
        wrapper.style.transform = "translateY(-10px) scale(0.95)";
        wrapper.style.filter = "blur(4px)";
        setTimeout(() => wrapper.remove(), 400);
    });


}
