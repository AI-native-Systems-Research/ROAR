// Diversity analytics visualization (Q3)

let codeDiversityTab = 'early';  // 'early', 'importance', 'decile'
let codeEarlyGroupBy = '';
let codeEarlyProblems = [];
let codeDecileGroupBy = '';
let codeDecileProblems = [];

// Factor analysis state
let codeFactorsGroupBy = '';
let codeFactorsProblems = [];

// Available problems (fetched from API)
let availableCodeDiversityProblems = [];
let availableCodeFactorsProblems = [];

// Dirty state tracking for Apply buttons
let codeEarlyDirty = false;
let codeFactorsDirty = false;
let codeDecileDirty = false;

function markCodeEarlyDirty() {
    codeEarlyDirty = true;
    updateCodeEarlyApplyButton();
}

function updateCodeEarlyApplyButton() {
    const btn = document.getElementById('codeEarlyApplyBtn');
    if (btn) {
        btn.className = codeEarlyDirty ? 'btn-apply dirty' : 'btn-apply';
        btn.textContent = codeEarlyDirty ? 'Apply *' : 'Apply';
    }
}

function markCodeFactorsDirty() {
    codeFactorsDirty = true;
    updateCodeFactorsApplyButton();
}

function updateCodeFactorsApplyButton() {
    const btn = document.getElementById('codeFactorsApplyBtn');
    if (btn) {
        btn.className = codeFactorsDirty ? 'btn-apply dirty' : 'btn-apply';
        btn.textContent = codeFactorsDirty ? 'Apply *' : 'Apply';
    }
}

function markCodeDecileDirty() {
    codeDecileDirty = true;
    updateCodeDecileApplyButton();
}

function updateCodeDecileApplyButton() {
    const btn = document.getElementById('codeDecileApplyBtn');
    if (btn) {
        btn.className = codeDecileDirty ? 'btn-apply dirty' : 'btn-apply';
        btn.textContent = codeDecileDirty ? 'Apply *' : 'Apply';
    }
}

function renderCodeDiversity() {
    const analyticsContent = document.getElementById('analyticsContent');

    let html = `
        <div class="diversity-description">
            <p><strong>Q3: Diversity Analysis</strong></p>
            <p>Measures code diversity using direct embeddings from jina-embeddings-v2-base-code (768-dim) applied to raw candidate code.</p>
        </div>

        <div class="diversity-tabs">
            <button class="tab-btn ${codeDiversityTab === 'early' ? 'active' : ''}" onclick="switchCodeDiversityTab('early')">
                Early Diversity
            </button>
            <button class="tab-btn ${codeDiversityTab === 'importance' ? 'active' : ''}" onclick="switchCodeDiversityTab('importance')">
                Factor Importance
            </button>
            <button class="tab-btn ${codeDiversityTab === 'decile' ? 'active' : ''}" onclick="switchCodeDiversityTab('decile')">
                Decile
            </button>
        </div>

        <div id="codeDiversityContent">
            <div class="loading">Loading...</div>
        </div>
    `;

    analyticsContent.innerHTML = html;
    loadCodeDiversityTab(codeDiversityTab);
}

async function switchCodeDiversityTab(tab) {
    codeDiversityTab = tab;

    // Update tab buttons
    document.querySelectorAll('.diversity-tabs .tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');

    loadCodeDiversityTab(tab);
}

async function loadCodeDiversityTab(tab) {
    const content = document.getElementById('codeDiversityContent');
    content.innerHTML = '<div class="loading">Loading...</div>';

    try {
        // Fetch available problems if not already loaded
        if (availableCodeDiversityProblems.length === 0) {
            await loadCodeDiversityProblems();
        }

        if (tab === 'early') {
            renderCodeEarlyTab();
        } else if (tab === 'importance') {
            await loadCodeFactorsProblems();
        } else if (tab === 'decile') {
            renderCodeDecileTab();
        }
    } catch (error) {
        content.innerHTML = `<div class="message error" style="display:block;">Error: ${error.message}</div>`;
    }
}

async function loadCodeDiversityProblems() {
    try {
        const res = await fetch('/analytics/code-diversity/problems');
        if (!res.ok) throw new Error('Failed to fetch problems');
        const data = await res.json();
        availableCodeDiversityProblems = data.problems || [];
    } catch (error) {
        console.error('Error loading code diversity problems:', error);
        availableCodeDiversityProblems = [];
    }
}

// ===========================================================================
// Early Diversity Tab
// ===========================================================================

function renderCodeEarlyTab() {
    const content = document.getElementById('codeDiversityContent');

    const problemCheckboxes = availableCodeDiversityProblems.map(p => {
        const safeId = p.replace(/[^a-zA-Z0-9]/g, '_');
        return `
            <label style="margin-right: 10px;">
                <input type="checkbox" id="code_early_prob_${safeId}" onchange="markCodeEarlyDirty()" ${codeEarlyProblems.includes(p) ? 'checked' : ''}>
                ${escapeHtml(p)}
            </label>
        `;
    }).join('');

    content.innerHTML = `
        <h3>Does Early Code Diversity Predict Final Outcome?</h3>
        <p class="section-description">Tests whether code diversity in early iterations (first 25%) correlates with better final scores. Color encodes model, marker shape and line style encode algorithm. Model+Algorithm grouping uses a faceted layout with one panel per algorithm.</p>

        <div class="variance-figure-container">
            <div class="figure-controls">
                <select id="codeEarlyGroupBySelect" onchange="markCodeEarlyDirty()">
                    <option value="" ${codeEarlyGroupBy === '' ? 'selected' : ''}>No Grouping (Aggregated)</option>
                    <option value="algorithm" ${codeEarlyGroupBy === 'algorithm' ? 'selected' : ''}>Group by Algorithm</option>
                    <option value="model" ${codeEarlyGroupBy === 'model' ? 'selected' : ''}>Group by Model</option>
                    <option value="model_algorithm" ${codeEarlyGroupBy === 'model_algorithm' ? 'selected' : ''}>Group by Model + Algorithm</option>
                </select>
                <button id="codeEarlyApplyBtn" class="btn-apply" onclick="applyCodeEarlyChanges()">Apply</button>
                <button class="btn-secondary" onclick="downloadCodeEarlyFigure()">Download</button>
            </div>
            <div class="problem-checkboxes" style="margin: 10px 0;">
                <label style="margin-right: 15px; font-weight: bold;">Problems:</label>
                ${problemCheckboxes}
            </div>
            <div class="figure-wrapper" id="codeEarlyFigureWrapper">
                <div class="empty-state">Select one or more problems above and click Apply to view early code diversity analysis.</div>
            </div>
        </div>
    `;
}

function getCodeEarlyFigureUrl() {
    let url = '/analytics/code-diversity/early/figure';
    const params = [];

    if (codeEarlyGroupBy) {
        params.push(`group_by=${encodeURIComponent(codeEarlyGroupBy)}`);
    }

    const selectedProblems = getSelectedCodeEarlyProblems();
    if (selectedProblems.length > 0 && selectedProblems.length < availableCodeDiversityProblems.length) {
        params.push(`problems=${encodeURIComponent(selectedProblems.join(','))}`);
    }

    if (params.length > 0) {
        url += '?' + params.join('&');
    }
    return url;
}

function getSelectedCodeEarlyProblems() {
    const problems = [];
    for (const p of availableCodeDiversityProblems) {
        const safeId = p.replace(/[^a-zA-Z0-9]/g, '_');
        const checkbox = document.getElementById(`code_early_prob_${safeId}`);
        if (checkbox && checkbox.checked) {
            problems.push(p);
        }
    }
    return problems;
}

function applyCodeEarlyChanges() {
    codeEarlyGroupBy = document.getElementById('codeEarlyGroupBySelect').value;
    codeEarlyProblems = getSelectedCodeEarlyProblems();
    codeEarlyDirty = false;
    updateCodeEarlyApplyButton();

    const wrapper = document.getElementById('codeEarlyFigureWrapper');
    if (!wrapper) return;

    if (codeEarlyProblems.length === 0) {
        wrapper.innerHTML = '<div class="empty-state">Select one or more problems above and click Apply to view early code diversity analysis.</div>';
        return;
    }

    const figUrl = getCodeEarlyFigureUrl();
    wrapper.innerHTML = `<img id="codeEarlyFigure" src="${figUrl}" alt="Early Code Diversity vs Outcome" />`;
}

function downloadCodeEarlyFigure() {
    const groupSuffix = codeEarlyGroupBy ? `_by_${codeEarlyGroupBy}` : '';
    const selectedProblems = getSelectedCodeEarlyProblems();
    const problemSuffix = selectedProblems.length > 0 && selectedProblems.length < 3
        ? `_${selectedProblems.join('_').toLowerCase()}`
        : '';
    const filename = `early_code_diversity_vs_outcome${groupSuffix}${problemSuffix}.png`;
    downloadCodeFigure(getCodeEarlyFigureUrl(), filename);
}

// ===========================================================================
// Factor Importance Tab
// ===========================================================================

async function loadCodeFactorsProblems() {
    try {
        const res = await fetch('/analytics/code-factors/problems');
        if (!res.ok) throw new Error('Failed to fetch problems');
        const data = await res.json();
        availableCodeFactorsProblems = data.problems || [];
        renderCodeFactorsControls();
    } catch (error) {
        const content = document.getElementById('codeDiversityContent');
        content.innerHTML = `<div class="message error" style="display:block;">Error loading problems: ${error.message}</div>`;
    }
}

function renderCodeFactorsControls() {
    const content = document.getElementById('codeDiversityContent');

    let tabTitle = 'Which Factors Matter Most for Score Improvement?';
    let tabDescription = 'Bar chart showing normalized factor importance for predicting new global bests. Values are z-score differences between improvement and non-improvement iterations.';

    content.innerHTML = `
        <h3>${tabTitle}</h3>
        <div class="variance-figure-container">
            <div class="figure-controls">
                <select id="codeFactorsGroupBySelect" onchange="markCodeFactorsDirty()">
                    <option value="" ${codeFactorsGroupBy === '' ? 'selected' : ''}>Aggregate</option>
                    <option value="model" ${codeFactorsGroupBy === 'model' ? 'selected' : ''}>Group by Model</option>
                    <option value="algorithm" ${codeFactorsGroupBy === 'algorithm' ? 'selected' : ''}>Group by Algorithm</option>
                    <option value="model_algorithm" ${codeFactorsGroupBy === 'model_algorithm' ? 'selected' : ''}>Group by Model + Algorithm</option>
                </select>
                <button id="codeFactorsApplyBtn" class="btn-apply" onclick="applyCodeFactorsChanges()">Apply</button>
                <button class="btn-secondary" onclick="downloadCodeFactorsFigure()">Download</button>
            </div>
            <p class="section-description">${tabDescription}</p>
            <div class="problem-checkboxes" style="margin: 10px 0;">
                <label style="margin-right: 15px; font-weight: bold;">Problems:</label>
                ${availableCodeFactorsProblems.map(p => `
                    <label style="margin-right: 10px;">
                        <input type="checkbox" id="code_factors_prob_${p.replace(/[^a-zA-Z0-9]/g, '_')}"
                               onchange="markCodeFactorsDirty()"
                               ${codeFactorsProblems.includes(p) ? 'checked' : ''}>
                        ${escapeHtml(p)}
                    </label>
                `).join('')}
            </div>
            <div class="figure-wrapper" id="codeFactorsFigureWrapper">
                <div class="empty-state">Select one or more problems above and click Apply to view factor analysis.</div>
            </div>
        </div>
    `;
}

function getSelectedCodeFactorsProblems() {
    const problems = [];
    for (const p of availableCodeFactorsProblems) {
        const checkbox = document.getElementById(`code_factors_prob_${p.replace(/[^a-zA-Z0-9]/g, '_')}`);
        if (checkbox && checkbox.checked) {
            problems.push(p);
        }
    }
    return problems;
}

function applyCodeFactorsChanges() {
    codeFactorsGroupBy = document.getElementById('codeFactorsGroupBySelect')?.value || '';
    codeFactorsProblems = getSelectedCodeFactorsProblems();

    codeFactorsDirty = false;
    updateCodeFactorsApplyButton();

    const wrapper = document.getElementById('codeFactorsFigureWrapper');
    if (!wrapper) return;

    if (codeFactorsProblems.length === 0) {
        wrapper.innerHTML = '<div class="empty-state">Select one or more problems above and click Apply to view factor analysis.</div>';
        return;
    }

    const figUrl = getCodeFactorsFigureUrl();
    wrapper.innerHTML = `
        <img id="codeFactorsFigure" src="${figUrl}" alt="Factor Analysis"
             onerror="this.parentElement.innerHTML='<div class=\\'empty-state\\'>Failed to load figure</div>'" />
    `;
}

function getCodeFactorsFigureUrl() {
    const params = [];

    if (codeFactorsGroupBy) {
        params.push(`group_by=${encodeURIComponent(codeFactorsGroupBy)}`);
    }

    if (codeFactorsProblems.length > 0) {
        params.push(`problems=${encodeURIComponent(codeFactorsProblems.join(','))}`);
    }

    let baseUrl = '/analytics/code-factors/importance/figure';

    if (params.length > 0) {
        return baseUrl + '?' + params.join('&');
    }
    return baseUrl;
}

function downloadCodeFactorsFigure() {
    if (codeFactorsProblems.length === 0) return;

    const link = document.createElement('a');
    link.href = getCodeFactorsFigureUrl();

    const groupSuffix = codeFactorsGroupBy ? `_by_${codeFactorsGroupBy}` : '';
    const problemSuffix = codeFactorsProblems.length < 3
        ? `_${codeFactorsProblems.join('_').replace(/[^a-zA-Z0-9_]/g, '').substring(0, 30)}`
        : '';

    link.download = `q3_factor_importance${groupSuffix}${problemSuffix}.png`;
    link.click();
}

// ===========================================================================
// Decile Diversity Tab
// ===========================================================================

function renderCodeDecileTab() {
    const content = document.getElementById('codeDiversityContent');

    const problemCheckboxes = availableCodeDiversityProblems.map(p => {
        const safeId = p.replace(/[^a-zA-Z0-9]/g, '_');
        return `
            <label style="margin-right: 10px;">
                <input type="checkbox" id="code_decile_prob_${safeId}" onchange="markCodeDecileDirty()" ${codeDecileProblems.includes(p) ? 'checked' : ''}>
                ${escapeHtml(p)}
            </label>
        `;
    }).join('');

    content.innerHTML = `
        <h3>Within-Decile Code Diversity Across Score Deciles</h3>
        <p class="section-description">Shows how code diversity varies across score deciles (1=lowest, 10=highest). Measures within-decile diversity to understand whether higher-scoring candidates are more or less diverse.</p>

        <div class="variance-figure-container">
            <div class="figure-controls">
                <select id="codeDecileGroupBySelect" onchange="markCodeDecileDirty()">
                    <option value="" ${codeDecileGroupBy === '' ? 'selected' : ''}>No Grouping (By Problem)</option>
                    <option value="algorithm" ${codeDecileGroupBy === 'algorithm' ? 'selected' : ''}>Group by Algorithm</option>
                    <option value="model" ${codeDecileGroupBy === 'model' ? 'selected' : ''}>Group by Model</option>
                    <option value="model_algorithm" ${codeDecileGroupBy === 'model_algorithm' ? 'selected' : ''}>Group by Model + Algorithm</option>
                </select>
                <button id="codeDecileApplyBtn" class="btn-apply" onclick="applyCodeDecileChanges()">Apply</button>
                <button class="btn-secondary" onclick="downloadCodeDecileFigure()">Download</button>
            </div>
            <div class="problem-checkboxes" style="margin: 10px 0;">
                <label style="margin-right: 15px; font-weight: bold;">Problems:</label>
                ${problemCheckboxes}
            </div>
            <div class="figure-wrapper" id="codeDecileFigureWrapper">
                <div class="empty-state">Select one or more problems above and click Apply to view decile diversity analysis.</div>
            </div>
        </div>
    `;
}

function getCodeDecileFigureUrl() {
    let url = '/analytics/code-diversity/decile/figure';
    const params = [];

    if (codeDecileGroupBy) {
        params.push(`group_by=${encodeURIComponent(codeDecileGroupBy)}`);
    }

    const selectedProblems = getSelectedCodeDecileProblems();
    if (selectedProblems.length > 0 && selectedProblems.length < availableCodeDiversityProblems.length) {
        params.push(`problems=${encodeURIComponent(selectedProblems.join(','))}`);
    }

    if (params.length > 0) {
        url += '?' + params.join('&');
    }
    return url;
}

function getSelectedCodeDecileProblems() {
    const problems = [];
    for (const p of availableCodeDiversityProblems) {
        const safeId = p.replace(/[^a-zA-Z0-9]/g, '_');
        const checkbox = document.getElementById(`code_decile_prob_${safeId}`);
        if (checkbox && checkbox.checked) {
            problems.push(p);
        }
    }
    return problems;
}

function applyCodeDecileChanges() {
    codeDecileGroupBy = document.getElementById('codeDecileGroupBySelect').value;
    codeDecileProblems = getSelectedCodeDecileProblems();
    codeDecileDirty = false;
    updateCodeDecileApplyButton();

    const wrapper = document.getElementById('codeDecileFigureWrapper');
    if (!wrapper) return;

    if (codeDecileProblems.length === 0) {
        wrapper.innerHTML = '<div class="empty-state">Select one or more problems above and click Apply to view decile diversity analysis.</div>';
        return;
    }

    const figUrl = getCodeDecileFigureUrl();
    wrapper.innerHTML = `<img id="codeDecileFigure" src="${figUrl}" alt="Decile Code Diversity" />`;
}

function downloadCodeDecileFigure() {
    const groupSuffix = codeDecileGroupBy ? `_by_${codeDecileGroupBy}` : '';
    const selectedProblems = getSelectedCodeDecileProblems();
    const problemSuffix = selectedProblems.length > 0 && selectedProblems.length < 3
        ? `_${selectedProblems.join('_').toLowerCase()}`
        : '';
    const filename = `decile_code_diversity${groupSuffix}${problemSuffix}.png`;
    downloadCodeFigure(getCodeDecileFigureUrl(), filename);
}

// ===========================================================================
// Utility Functions
// ===========================================================================

function downloadCodeFigure(url, filename) {
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.click();
}
