/* Shared behaviour for the LiteratureDB filter (left) and overview (right) panels.
 *
 * Pages call initLiteraturePanels({ overviewUrl, initialOverview, activeFilters, onChange })
 * once the DOM is ready. Requires Plotly to be loaded first (charts are skipped
 * gracefully if it is missing).
 */
(function () {
    "use strict";

    var HAS_PLOTLY = typeof Plotly !== "undefined";
    var PLOT_CFG = { displayModeBar: false, responsive: true };
    var PLOT_BASE = { margin: { t: 8, r: 8, b: 34, l: 40 }, font: { size: 10 }, showlegend: false };

    function escapeHtml(s) {
        return String(s).replace(/[&<>"']/g, function (c) {
            return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
        });
    }
    function truncate(s, n) { return s.length > n ? s.slice(0, n - 1) + "…" : s; }

    function currentFilters() {
        var out = {};
        document.querySelectorAll('#filterForm input[type="checkbox"]:checked').forEach(function (cb) {
            (out[cb.name] = out[cb.name] || []).push(cb.value);
        });
        return out;
    }
    window.currentLiteratureFilters = currentFilters;

    function renderPaperList(data) {
        var countEl = document.getElementById("paperCount");
        if (countEl) countEl.textContent = data.total.toLocaleString();

        var items = document.getElementById("paperListItems");
        if (!items) return;
        items.innerHTML = "";
        data.papers.forEach(function (p) {
            var li = document.createElement("li");
            var meta = [p.first_author, p.year, p.journal].filter(Boolean).join(", ");
            var title = p.openalex_id
                ? '<a href="' + escapeHtml(p.openalex_id) + '" target="_blank" rel="noopener">' + escapeHtml(p.title) + "</a>"
                : escapeHtml(p.title);
            li.innerHTML = title + (meta ? ' <span class="meta">(' + escapeHtml(meta) + ")</span>" : "");
            items.appendChild(li);
        });
        var note = document.getElementById("paperListNote");
        if (note) {
            note.textContent = data.total > data.paper_limit
                ? "Showing the " + data.paper_limit + " most recent of " + data.total.toLocaleString() + " papers."
                : "Showing all papers in the current selection.";
        }
    }

    function renderOverview(data) {
        var totalEl = document.getElementById("ovTotal");
        if (totalEl) totalEl.textContent = data.total.toLocaleString();
        renderPaperList(data);

        if (!HAS_PLOTLY) return;

        Plotly.react("chartYear",
            [{ type: "bar", x: data.by_year.map(function (d) { return d[0]; }),
               y: data.by_year.map(function (d) { return d[1]; }), marker: { color: "#1565c0" } }],
            Object.assign({}, PLOT_BASE), PLOT_CFG);

        var jr = data.by_journal.slice().reverse();
        Plotly.react("chartJournal",
            [{ type: "bar", orientation: "h", x: jr.map(function (d) { return d[1]; }),
               y: jr.map(function (d) { return truncate(d[0], 28); }), marker: { color: "#1565c0" } }],
            Object.assign({}, PLOT_BASE, { margin: { t: 8, r: 8, b: 34, l: 150 } }), PLOT_CFG);

        Plotly.react("chartCountry",
            [{ type: "choropleth", locationmode: "ISO-3",
               locations: data.by_country.map(function (d) { return d[0]; }),
               z: data.by_country.map(function (d) { return d[1]; }),
               colorscale: "Blues", showscale: false }],
            { margin: { t: 0, r: 0, b: 0, l: 0 }, font: { size: 10 },
              geo: { showframe: false, showcoastlines: false, projection: { type: "natural earth" } } },
            PLOT_CFG);
    }
    window.renderLiteratureOverview = renderOverview;

    window.initLiteraturePanels = function (opts) {
        opts = opts || {};
        var form = document.getElementById("filterForm");

        if (!HAS_PLOTLY) {
            var miss = document.getElementById("plotlyMissing");
            if (miss) miss.style.display = "block";
        }

        // Pre-check the boxes for the filters that are already active.
        var active = opts.activeFilters || {};
        if (form) {
            Object.keys(active).forEach(function (group) {
                (active[group] || []).forEach(function (val) {
                    form.querySelectorAll('input[name="' + group + '"]').forEach(function (cb) {
                        if (cb.value === val) cb.checked = true;
                    });
                });
            });
        }

        function refresh() {
            if (typeof opts.onChange === "function") opts.onChange(currentFilters());
            if (!opts.overviewUrl) return;
            var qs = encodeURIComponent(JSON.stringify(currentFilters()));
            fetch(opts.overviewUrl + "?filters=" + qs)
                .then(function (r) { return r.json(); })
                .then(renderOverview)
                .catch(function (e) { console.error("overview fetch failed", e); });
        }
        window.refreshLiteratureOverview = refresh;

        if (form) {
            form.addEventListener("submit", function (e) { e.preventDefault(); refresh(); });
            var clear = document.getElementById("clearFilters");
            if (clear) clear.addEventListener("click", function () { form.reset(); refresh(); });
        }

        if (typeof opts.onChange === "function") opts.onChange(currentFilters());
        if (opts.initialOverview) renderOverview(opts.initialOverview);
    };
})();
