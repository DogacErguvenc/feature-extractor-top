import React, { useEffect, useMemo, useState } from "react";
import axios from "axios";
import { API } from "@/App";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { toast } from "sonner";
import "./ResnetAnalysis.css";

const formatPercent = (value) => {
  const num = Number(value);
  if (!Number.isFinite(num)) return "-";
  return `%${num.toFixed(1)}`;
};

const formatScore = (value) => {
  const num = Number(value);
  if (!Number.isFinite(num)) return "-";
  return `${num.toFixed(1)}%`;
};

const formatTimestamp = (value) => {
  if (!value) return "-";
  try {
    return new Date(value).toLocaleString("tr-TR");
  } catch {
    return String(value);
  }
};

const toInt = (value, fallback) => {
  const num = Number(value);
  return Number.isFinite(num) ? Math.floor(num) : fallback;
};

const MAX_SAMPLE_LIMIT = 5000;
const SAMPLE_RENDER_CHUNK = 200;

const ResnetAnalysis = () => {
  const [loadingOverview, setLoadingOverview] = useState(true);
  const [loadingDetails, setLoadingDetails] = useState(false);
  const [overview, setOverview] = useState(null);
  const [perPlu, setPerPlu] = useState([]);
  const [selectedPlu, setSelectedPlu] = useState("");
  const [selectedStats, setSelectedStats] = useState(null);
  const [samples, setSamples] = useState([]);
  const [pluSearch, setPluSearch] = useState("");
  const [thresholdInput, setThresholdInput] = useState("35");
  const [threshold, setThreshold] = useState(35);
  const [sampleLimitInput, setSampleLimitInput] = useState("60");
  const [sampleLimit, setSampleLimit] = useState(60);
  const [onlyLowConf, setOnlyLowConf] = useState(false);
  const [renderedSampleCount, setRenderedSampleCount] = useState(SAMPLE_RENDER_CHUNK);

  const fetchOverview = async (thresholdValue = threshold, refreshOverview = false) => {
    setLoadingOverview(true);
    try {
      const res = await axios.get(`${API}/stats/resnet-top5-analysis`, {
        params: {
          low_conf_threshold_pct: thresholdValue,
          sample_limit: 1,
          refresh: refreshOverview,
        },
      });
      const payload = res.data || {};
      setOverview(payload.summary || null);
      const rows = payload.per_plu || [];
      setPerPlu(rows);
      if (!selectedPlu && rows.length > 0) {
        setSelectedPlu(rows[0].plu_code);
      } else if (selectedPlu && rows.length > 0 && !rows.some((r) => r.plu_code === selectedPlu)) {
        setSelectedPlu(rows[0].plu_code);
      }
    } catch (error) {
      console.error("resnet overview error", error);
      toast.error("ResNet analiz ozeti yuklenemedi");
    } finally {
      setLoadingOverview(false);
    }
  };

  const fetchSelectedPlu = async (pluCode, thresholdValue = threshold, onlyLow = onlyLowConf, limit = sampleLimit) => {
    if (!pluCode) {
      setSelectedStats(null);
      setSamples([]);
      return;
    }

    setLoadingDetails(true);
    try {
      const res = await axios.get(`${API}/stats/resnet-top5-analysis`, {
        params: {
          plu_code: pluCode,
          low_conf_threshold_pct: thresholdValue,
          sample_limit: limit,
          include_only_low_conf: onlyLow,
          include_overview: false,
        },
      });
      const payload = res.data || {};
      const nextSamples = payload.samples || [];
      if (payload.summary) {
        setOverview(payload.summary);
      }
      setSelectedStats(payload.selected_plu || null);
      setSamples(nextSamples);
      setRenderedSampleCount(Math.min(nextSamples.length, SAMPLE_RENDER_CHUNK));
    } catch (error) {
      console.error("resnet detail error", error);
      toast.error("PLU analiz detaylari yuklenemedi");
    } finally {
      setLoadingDetails(false);
    }
  };

  useEffect(() => {
    fetchOverview();
  }, []);

  useEffect(() => {
    if (selectedPlu) {
      fetchSelectedPlu(selectedPlu);
    }
  }, [selectedPlu]);

  const filteredPlu = useMemo(() => {
    const query = String(pluSearch || "").trim().toLowerCase();
    if (!query) return perPlu;
    return perPlu.filter((row) => {
      const code = String(row.plu_code || "").toLowerCase();
      return code.includes(query);
    });
  }, [perPlu, pluSearch]);
  const visibleSamples = useMemo(
    () => samples.slice(0, Math.max(0, renderedSampleCount)),
    [samples, renderedSampleCount]
  );

  const handleApplyFilters = async () => {
    const rawThreshold = Number(thresholdInput);
    const nextThreshold = Number.isFinite(rawThreshold)
      ? Math.max(0, Math.min(100, rawThreshold))
      : 35;
    const nextLimit = Math.max(1, Math.min(MAX_SAMPLE_LIMIT, toInt(sampleLimitInput, 60)));
    setThreshold(nextThreshold);
    setSampleLimit(nextLimit);
    setThresholdInput(String(nextThreshold));
    setSampleLimitInput(String(nextLimit));

    const shouldRefreshOverview = nextThreshold !== threshold;
    if (shouldRefreshOverview) {
      await Promise.all([
        fetchOverview(nextThreshold),
        fetchSelectedPlu(selectedPlu, nextThreshold, onlyLowConf, nextLimit),
      ]);
      return;
    }
    await fetchSelectedPlu(selectedPlu, nextThreshold, onlyLowConf, nextLimit);
  };

  const handleRefresh = async () => {
    await Promise.all([
      fetchOverview(threshold, true),
      fetchSelectedPlu(selectedPlu, threshold, onlyLowConf, sampleLimit),
    ]);
  };

  const loading = loadingOverview || loadingDetails;

  return (
    <div className="page-container resnet-analysis-page" data-testid="resnet-analysis-page">
      <div className="resnet-analysis-header">
        <div>
          <h1 data-testid="resnet-analysis-title">ResNet Top-5 Analiz</h1>
          <p className="subtitle">
            Dashboard disinda, PLU bazli kompakt analiz ekrani.
          </p>
        </div>
        <div className="resnet-header-actions">
          <Button variant="outline" onClick={handleRefresh} disabled={loading}>
            Yenile
          </Button>
        </div>
      </div>

      <div className="resnet-summary-grid">
        <Card className="resnet-stat-card">
          <CardHeader>
            <CardDescription>Toplam ResNet Kaydi</CardDescription>
            <CardTitle>{overview?.total ?? 0}</CardTitle>
          </CardHeader>
        </Card>
        <Card className="resnet-stat-card">
          <CardHeader>
            <CardDescription>Top-5 Uyumlu</CardDescription>
            <CardTitle>
              {overview?.top5_match_count ?? 0} ({formatPercent(overview?.top5_match_rate)})
            </CardTitle>
          </CardHeader>
        </Card>
        <Card className="resnet-stat-card">
          <CardHeader>
            <CardDescription>Top-5 Uyumsuz</CardDescription>
            <CardTitle>{overview?.top5_mismatch_count ?? 0}</CardTitle>
          </CardHeader>
        </Card>
        <Card className="resnet-stat-card">
          <CardHeader>
            <CardDescription>Dusuk Skor Top-5</CardDescription>
            <CardTitle>
              {overview?.low_conf_top5_count ?? 0} ({formatPercent(overview?.low_conf_top5_rate)})
            </CardTitle>
          </CardHeader>
        </Card>
      </div>

      <div className="resnet-main-grid">
        <Card className="resnet-plu-card">
          <CardHeader>
            <CardTitle>PLU Listesi</CardTitle>
            <CardDescription>Toplam: {perPlu.length} PLU</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="resnet-plu-search">
              <input
                type="text"
                placeholder="PLU ara..."
                value={pluSearch}
                onChange={(e) => setPluSearch(e.target.value)}
              />
            </div>
            <div className="resnet-plu-list">
              {filteredPlu.length === 0 && <p className="muted">PLU bulunamadi.</p>}
              {filteredPlu.map((row) => {
                const active = row.plu_code === selectedPlu;
                return (
                  <button
                    type="button"
                    key={row.plu_code}
                    className={`resnet-plu-row ${active ? "active" : ""}`}
                    onClick={() => setSelectedPlu(row.plu_code)}
                  >
                    <div className="resnet-plu-row-main">
                      <strong>{row.plu_code}</strong>
                      <span>{row.total} kayit</span>
                    </div>
                    <div className="resnet-plu-row-sub">
                      <span>Top-5: {formatPercent(row.top5_match_rate)}</span>
                      <span>Dusuk skor: {row.low_conf_top5_count}</span>
                    </div>
                  </button>
                );
              })}
            </div>
          </CardContent>
        </Card>

        <div className="resnet-detail-column">
          <Card className="resnet-filter-card">
            <CardHeader>
              <CardTitle>Filtreler</CardTitle>
              <CardDescription>Secili PLU: {selectedPlu || "-"}</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="resnet-filter-grid">
                <label>
                  Dusuk skor esigi (%)
                  <input
                    type="number"
                    min="0"
                    max="100"
                    value={thresholdInput}
                    onChange={(e) => setThresholdInput(e.target.value)}
                  />
                </label>
                <label>
                  Liste limiti
                  <input
                    type="number"
                    min="1"
                    max={MAX_SAMPLE_LIMIT}
                    value={sampleLimitInput}
                    onChange={(e) => setSampleLimitInput(e.target.value)}
                  />
                </label>
              </div>
              <label className="resnet-checkbox">
                <input
                  type="checkbox"
                  checked={onlyLowConf}
                  onChange={(e) => setOnlyLowConf(e.target.checked)}
                />
                Sadece dusuk skorlu Top-5 uyumlu kayitlari goster
              </label>
              <div className="resnet-filter-actions">
                <Button onClick={handleApplyFilters} disabled={loading}>
                  Filtreyi Uygula
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card className="resnet-selected-card">
            <CardHeader>
              <CardTitle>Secili PLU Ozet</CardTitle>
              <CardDescription>
                Uyumlu/Uyumsuz degerleri ResNet Top-5 sonucuna gore hesaplanir.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {!selectedStats ? (
                <p className="muted">Bir PLU secin.</p>
              ) : (
                <div className="resnet-selected-grid">
                  <div>
                    <p className="k">Toplam</p>
                    <p className="v">{selectedStats.total}</p>
                  </div>
                  <div>
                    <p className="k">Top-5 Uyumlu</p>
                    <p className="v">
                      {selectedStats.top5_match_count} ({formatPercent(selectedStats.top5_match_rate)})
                    </p>
                  </div>
                  <div>
                    <p className="k">Top-5 Uyumsuz</p>
                    <p className="v">{selectedStats.top5_mismatch_count}</p>
                  </div>
                  <div>
                    <p className="k">Top-1 Uyumlu</p>
                    <p className="v">
                      {selectedStats.top1_match_count} ({formatPercent(selectedStats.top1_match_rate)})
                    </p>
                  </div>
                  <div>
                    <p className="k">Dusuk Skor Top-5</p>
                    <p className="v">
                      {selectedStats.low_conf_top5_count} ({formatPercent(selectedStats.low_conf_top5_rate)})
                    </p>
                  </div>
                  <div>
                    <p className="k">Ort. Secili PLU Skoru</p>
                    <p className="v">{formatScore(selectedStats.avg_selected_score_pct)}</p>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          <Card className="resnet-samples-card">
            <CardHeader>
              <CardTitle>Ornek Kayitlar</CardTitle>
              <CardDescription>
                Toplam {samples.length} kayittan {visibleSamples.length} gosteriliyor.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {samples.length === 0 ? (
                <p className="muted">Filtreye uygun kayit yok.</p>
              ) : (
                <div className="resnet-sample-list">
                  {visibleSamples.map((row) => (
                    <div className="resnet-sample-row" key={row.validation_id || `${row.timestamp}-${row.plu_code}`}>
                      <div className="resnet-sample-main">
                        <p className="file">{row.filename || "-"}</p>
                        <p className="meta">{formatTimestamp(row.timestamp)}</p>
                        <p className="meta">Top-5: {(row.top5_codes || []).join(", ") || "-"}</p>
                      </div>
                      <div className="resnet-sample-stats">
                        <span className={`chip ${row.is_top5_match ? "ok" : "bad"}`}>
                          {row.is_top5_match ? "Top-5 Uyumlu" : "Top-5 Uyumsuz"}
                        </span>
                        {row.is_low_conf_top5 && <span className="chip warn">Dusuk Skor</span>}
                        <span>Secili skor: {formatScore(row.selected_score_pct)}</span>
                        <span>Top-1 skor: {formatScore(row.top1_score_pct)}</span>
                        <span>Tahmin: {row.predicted_plu || "-"}</span>
                      </div>
                    </div>
                  ))}
                  {visibleSamples.length < samples.length && (
                    <div className="resnet-sample-actions">
                      <Button
                        variant="outline"
                        onClick={() =>
                          setRenderedSampleCount((current) =>
                            Math.min(samples.length, current + SAMPLE_RENDER_CHUNK)
                          )
                        }
                      >
                        Daha Fazla Goster (+{SAMPLE_RENDER_CHUNK})
                      </Button>
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default ResnetAnalysis;
