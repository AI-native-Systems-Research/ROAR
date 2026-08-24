// Summary analytics visualization

function renderSummary(data) {
    const analyticsContent = document.getElementById('analyticsContent');

    let html = `
        <div class="stats-grid">
            <div class="stat-card">
                <div class="value">${data.campaign_count}</div>
                <div class="label">Campaigns</div>
            </div>
            <div class="stat-card">
                <div class="value">${data.iteration_count}</div>
                <div class="label">Iterations</div>
            </div>
            <div class="stat-card">
                <div class="value">${data.candidate_count}</div>
                <div class="label">Candidates</div>
            </div>
            <div class="stat-card">
                <div class="value">${data.model_count}</div>
                <div class="label">Models</div>
            </div>
            <div class="stat-card">
                <div class="value">${data.algorithm_count}</div>
                <div class="label">Algorithms</div>
            </div>
        </div>
    `;

    if (data.models && data.models.length > 0) {
        html += `
            <div class="list-section">
                <h3>Models Used</h3>
                <div class="tag-list">
                    ${data.models.map(m => `<span class="tag">${escapeHtml(m)}</span>`).join('')}
                </div>
            </div>
        `;
    }

    if (data.algorithms && data.algorithms.length > 0) {
        html += `
            <div class="list-section">
                <h3>Algorithms</h3>
                <div class="tag-list">
                    ${data.algorithms.map(a => `<span class="tag">${escapeHtml(a)}</span>`).join('')}
                </div>
            </div>
        `;
    }

    if (data.campaign_count === 0) {
        html = '<div class="empty-state">No campaigns yet. Upload some data to see analytics.</div>';
    }

    analyticsContent.innerHTML = html;
}
