document.addEventListener("DOMContentLoaded", function() {
    let currentSlide = 0;
    const slides = document.querySelectorAll(".slide");

    function showSlide(index) {
        slides.forEach(slide => slide.style.display = "none");
        slides[index].style.display = "block";
    }

    function nextSlide() {
        currentSlide = (currentSlide + 1) % slides.length;
        showSlide(currentSlide);
    }

    showSlide(currentSlide);
    setInterval(nextSlide, 3000);
});

function showSection(sectionId) {
    document.querySelectorAll(".section").forEach(section => section.classList.remove("visible"));
    document.getElementById(sectionId).classList.add("visible");
}

document.getElementById("toggleMode").addEventListener("click", function() {
    document.body.classList.toggle("dark-mode");
});

document.getElementById("predictBtn").addEventListener("click", function() {
    const stockInput = document.getElementById("stockInput").value.trim().toUpperCase();
    const predictionResult = document.getElementById("predictionResult");

    const samplePredictions = {
        "ESABINDIA": `Upcoming Predicted Stock Unit Prices: [2566.6987 2637.3677 2709.3833 2732.658  2734.0867 2733.7969 2749.879 2750.7932 2734.4421 2714.8342]<br><b>Trend:</b> Upward with minor dips.<br><b>Recommendation:</b> Look for dips to enter.`,
        "TCS": `Upcoming Predicted Stock Unit Prices: [2907.5078 2917.679  2924.523  2936.3945 2959.3276 2997.2375 3023.6501 3023.8206 3025.037  3049.7842]<br><b>Trend:</b> Upward.<br><b>Recommendation:</b> Buy for long-term growth.`,
        "INFY": `Upcoming Predicted Stock Unit Prices: [1476.0575 1481.7906 1485.6517 1484.8787 1486.6354 1486.2185 1474.5215 1448.477  1427.641  1422.8749]<br><b>Trend:</b> Downward.<br><b>Recommendation:</b> Caution advised, wait for stability.`,
        "HDFC": `Upcoming Predicted Stock Unit Prices: [1984.0519 1998.8048 2008.7761 2021.2169 2041.1055 2067.775  2092.8035 2093.3457 2077.6848 2072.5159]<br><b>Trend:</b> Volatile but upward.<br><b>Recommendation:</b> Consider long-term buy.`,
        "RELIANCE": `Upcoming Predicted Stock Unit Prices: [1863.5934 1877.7343 1878.2699 1874.7227 1874.977  1890.7026 1911.3578 1916.3092 1918.4341 1918.736]<br><b>Trend:</b> Slight upward.<br><b>Recommendation:</b> Buy and hold for steady gains.`,
        "ASIANPAINT": `Upcoming Predicted Stock Unit Prices: [2575.8218 2605.8386 2631.237  2653.033  2674.387  2698.5278 2730.9453 2752.2668 2777.4946 2797.805]<br><b>Trend:</b> Consistent upward.<br><b>Recommendation:</b> Buy and hold.`,
        "AMBER": `Upcoming Predicted Stock Unit Prices: [2631.752  2620.207  2605.7368 2591.9404 2582.8552 2582.734  2607.734 2649.4966 2684.2742 2718.9402]<br><b>Trend:</b> Recovering after dip.<br><b>Recommendation:</b> Buy at dip (around 2582).`,
        "BARBEQUE": `Upcoming Predicted Stock Unit Prices: [945.5205  987.2679 1021.503  1069.6699 1110.3499 1128.3859 1134.5945 1125.6421 1126.482  1121.1661]<br><b>Trend:</b> Initial rise, then plateau.<br><b>Recommendation:</b> Buy now, sell before plateau.`
    };

    predictionResult.innerHTML = samplePredictions[stockInput] ?
        `<h3>Predicted Analysis for <b>${stockInput}</b>:</h3><p>${samplePredictions[stockInput]}</p>` :
        `<p style="color:red;">No sample data available for "${stockInput}".</p>`;
});

document.getElementById("getInsightsBtn").addEventListener("click", function() {
    const insightsResult = document.getElementById("insightsResult");

    const insightsHTML = `<h3>Market Insights</h3>
<p><b>Profitable Stocks (Relatively Higher Value):</b></p>
<ul>
<li><b>ASIANPAINT</b>: Appears strong at 3594.54. Stable investment.</li>
<li><b>AMBER</b>: High at 3489.78. Growth in consumer durables likely.</li>
<li><b>ESABINDIA</b>: Valued at 3518.06. Strong performance.</li>
</ul>
<p><b>Risky Stocks (Lower Value & Higher Volatility):</b></p>
<ul>
<li><b>TRF, HDIL, CINEVISTA, OILCOUNTUB</b>: Penny stock risks. Research advised.</li>
<li><b>ABAN</b>: Low value at 123.60. High risk/reward potential.</li>
</ul>
<p><b>Actionable Recommendations:</b></p>
<ul>
<li>Invest more in stable stocks like ASIANPAINT, AMBER, and ESABINDIA.</li>
<li>Allocate smaller funds to risky options only with proper research.</li>
<li>Always conduct due diligence before investing based on a single data point.</li>
</ul>`;

    insightsResult.innerHTML = insightsHTML;
});