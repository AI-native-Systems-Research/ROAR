// Cluster analytics visualization

function renderClusters(data) {
    const analyticsContent = document.getElementById('analyticsContent');

    if (data.cluster_count === 0) {
        analyticsContent.innerHTML = '<div class="empty-state">No clusters found. Upload campaigns with research questions to see clusters.</div>';
        return;
    }

    const totalCampaigns = data.clusters.reduce((sum, c) => sum + c.campaign_count, 0);
    const maxCount = Math.max(...data.clusters.map(c => c.campaign_count));

    let html = `
        <div class="cluster-summary">
            <span class="cluster-summary-stat"><span class="value">${data.cluster_count}</span> clusters</span>
            <span class="cluster-summary-stat"><span class="value">${totalCampaigns}</span> campaigns clustered</span>
            ${data.campaigns_without_embeddings > 0 ? `<span class="cluster-summary-stat"><span class="value">${data.campaigns_without_embeddings}</span> without embeddings</span>` : ''}
        </div>
        <div class="cluster-chart">
    `;

    // Sort by campaign count descending
    const sortedClusters = [...data.clusters].sort((a, b) => b.campaign_count - a.campaign_count);

    for (const cluster of sortedClusters) {
        const representative = cluster.research_questions.length > 0
            ? cluster.research_questions[0]
            : 'Unknown';
        const barWidth = (cluster.campaign_count / maxCount) * 100;

        html += `
            <div class="cluster-bar">
                <div class="cluster-label" title="${escapeHtml(representative)}">${escapeHtml(representative)}</div>
                <div class="cluster-bar-container">
                    <div class="cluster-bar-fill" style="width: ${barWidth}%;"></div>
                    <span class="cluster-bar-value">${cluster.campaign_count}</span>
                </div>
            </div>
        `;
    }

    html += '</div>';
    analyticsContent.innerHTML = html;
}
