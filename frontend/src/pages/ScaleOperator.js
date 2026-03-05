import React, { useState, useEffect, useCallback, useRef } from "react";
import axios from "axios";
import { API } from "@/App";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { toast } from "sonner";
import "./ScaleOperator.css";

const formatTopMatchScore = (match) => {
  const raw = Number(match?.score);
  if (!Number.isFinite(raw)) return "-";
  if (match?.score_type === "cosine_similarity") {
    return raw.toFixed(4);
  }
  return `${raw.toFixed(1)}%`;
};

const summarizeTopMatches = (matches) => {
  if (!Array.isArray(matches) || matches.length === 0) return "-";
  return matches
    .slice(0, 3)
    .map((match, idx) => {
      const rank = match?.rank ?? idx + 1;
      const pluCode = match?.plu_code || "?";
      return `${rank}. ${pluCode} (${formatTopMatchScore(match)})`;
    })
    .join(" | ");
};

const ScaleOperator = () => {
  const [plusList, setPlusList] = useState([]);
  const [loading, setLoading] = useState(true);
  const [systemMode, setSystemMode] = useState("training");
  const [processing, setProcessing] = useState(false);
  const [selectedPLU, setSelectedPLU] = useState(null);

  const [liveRunning, setLiveRunning] = useState(false);
  const [livePLU, setLivePLU] = useState(null);
  const [liveResult, setLiveResult] = useState(null);
  const [liveLoading, setLiveLoading] = useState(false);
  const [liveError, setLiveError] = useState("");
  const liveRequestRef = useRef(false);

  useEffect(() => {
    fetchPLUList();
    fetchSystemMode();
  }, []);

  const fetchPLUList = async () => {
    try {
      const response = await axios.get(`${API}/plu/list`);
      setPlusList(response.data);
      setLoading(false);
    } catch (error) {
      console.error("Error fetching PLU list:", error);
      toast.error("PLU listesi yuklenemedi");
      setLoading(false);
    }
  };

  const fetchSystemMode = async () => {
    try {
      const response = await axios.get(`${API}/system/mode`);
      setSystemMode(response.data.mode);
    } catch (error) {
      console.error("Error fetching system mode:", error);
    }
  };

  const handlePLUSelect = async (plu) => {
    if (processing) return;

    setProcessing(true);
    setSelectedPLU(plu);

    try {
      const response = await axios.post(`${API}/plu/select`, {
        plu_code: plu.plu_code,
      });

      toast.success(`${plu.name} secildi ve fotograf cekildi`);

      if (systemMode === "training") {
        toast.info(`Toplanan fotograf sayisi: ${response.data.images_collected || 1}`);
      } else {
        toast.info("AI analizi baslatildi");
      }

      setTimeout(() => {
        setSelectedPLU(null);
      }, 2000);
    } catch (error) {
      console.error("Error selecting PLU:", error);
      toast.error("Hata: " + (error.response?.data?.detail || "Fotograf cekilemedi"));
      setSelectedPLU(null);
    } finally {
      setProcessing(false);
    }
  };

  const stopLiveMode = useCallback(() => {
    setLiveRunning(false);
    setLiveLoading(false);
    liveRequestRef.current = false;
  }, []);

  const fetchLiveResult = useCallback(async (plu) => {
    if (!plu || liveRequestRef.current) return;

    liveRequestRef.current = true;
    setLiveLoading(true);

    try {
      const response = await axios.post(`${API}/live/validate`, {
        plu_code: plu.plu_code,
        camera_index: 0,
        persist_capture: false,
        persist_validation: false,
      });

      setLiveResult(response.data);
      setLiveError("");
    } catch (error) {
      console.error("Error getting live result:", error);
      setLiveError(error.response?.data?.detail || "Canli sonuc alinamadi");
    } finally {
      liveRequestRef.current = false;
      setLiveLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!liveRunning || !livePLU) return undefined;

    let cancelled = false;
    const liveLoop = async () => {
      while (!cancelled) {
        await fetchLiveResult(livePLU);
        // Let the UI thread breathe while requesting the next frame immediately.
        await new Promise((resolve) => window.setTimeout(resolve, 0));
      }
    };
    liveLoop();

    return () => {
      cancelled = true;
    };
  }, [liveRunning, livePLU, fetchLiveResult]);

  useEffect(() => {
    return () => {
      liveRequestRef.current = false;
    };
  }, []);

  const toggleLiveMode = (plu) => {
    const samePLU = liveRunning && livePLU?.plu_code === plu.plu_code;
    if (samePLU) {
      stopLiveMode();
      return;
    }

    setLivePLU(plu);
    setLiveResult(null);
    setLiveError("");
    setLiveRunning(true);
  };

  if (loading) {
    return (
      <div className="page-container">
        <div className="spinner"></div>
      </div>
    );
  }

  const liveImageBase64 = liveResult?.processed_image_base64 || liveResult?.image_base64;
  const liveIsMatch = Boolean(liveResult?.is_match);

  return (
    <div className="page-container scale-operator" data-testid="scale-operator-page">
      <div className="operator-header">
        <div>
          <h1 data-testid="page-title">Terazi Operator Ekrani</h1>
          <p className="subtitle" data-testid="page-subtitle">Lutfen urun icin PLU secimi yapin</p>
        </div>
        <div className={`mode-badge ${systemMode}`} data-testid="system-mode-badge">
          <span className="mode-indicator"></span>
          {systemMode === "training" ? "Egitim Modu" : "Uretim Modu"}
        </div>
      </div>

      {plusList.length === 0 ? (
        <Card className="empty-state" data-testid="empty-plu-state">
          <CardContent className="empty-content">
            <div className="empty-icon">🏷️</div>
            <h3>Henuz PLU eklenmemis</h3>
            <p>Baslamak icin PLU Yonetimi sayfasindan urun ekleyin</p>
          </CardContent>
        </Card>
      ) : (
        <div className="plu-grid" data-testid="plu-grid">
          {plusList.map((plu) => {
            const isLiveTarget = liveRunning && livePLU?.plu_code === plu.plu_code;
            return (
              <Card
                key={plu.id}
                className={`plu-card ${selectedPLU?.id === plu.id ? "selected" : ""} ${processing && selectedPLU?.id === plu.id ? "processing" : ""} ${isLiveTarget ? "live-selected" : ""}`}
                data-testid={`plu-card-${plu.plu_code}`}
              >
                <CardHeader>
                  <CardTitle data-testid={`plu-title-${plu.plu_code}`}>
                    <span className="plu-code" data-testid={`plu-code-${plu.plu_code}`}>PLU {plu.plu_code}</span>
                  </CardTitle>
                  <CardDescription data-testid={`plu-name-${plu.plu_code}`}>{plu.name}</CardDescription>
                </CardHeader>
                <CardContent>
                  <p className="plu-description" data-testid={`plu-description-${plu.plu_code}`}>{plu.description}</p>
                  <Button
                    className="select-btn"
                    onClick={() => handlePLUSelect(plu)}
                    disabled={processing}
                    data-testid={`plu-select-btn-${plu.plu_code}`}
                  >
                    {processing && selectedPLU?.id === plu.id ? (
                      <span>Isleniyor...</span>
                    ) : (
                      <span>Sec</span>
                    )}
                  </Button>
                  <Button
                    className={`live-btn ${isLiveTarget ? "stop" : ""}`}
                    onClick={() => toggleLiveMode(plu)}
                    disabled={processing}
                    data-testid={`plu-live-btn-${plu.plu_code}`}
                  >
                    {isLiveTarget ? "Canliyi Durdur" : "Canli Tahmin"}
                  </Button>
                </CardContent>
              </Card>
            );
          })}
        </div>
      )}

      {livePLU && (
        <Card className="live-card" data-testid="live-result-card">
          <CardHeader>
            <CardTitle>Canli Sonuc Paneli</CardTitle>
            <CardDescription>
              PLU {livePLU.plu_code} - {livePLU.name} | Yeni kare onceki sonuc gelir gelmez alinir
            </CardDescription>
          </CardHeader>
          <CardContent className="live-content">
            <div className="live-status-row">
              <span className={`live-status ${liveRunning ? "active" : "idle"}`}>
                {liveRunning ? (liveLoading ? "Yeni kare aliniyor..." : "Canli izleme aktif") : "Canli izleme durduruldu"}
              </span>
              <Button
                variant="outline"
                onClick={stopLiveMode}
                disabled={!liveRunning}
                data-testid="live-stop-btn"
              >
                Durdur
              </Button>
            </div>

            {liveError && <p className="live-error">{liveError}</p>}

            {liveResult ? (
              <div className="live-result-grid">
                <div className="live-result-details">
                  <p className={`live-match ${liveIsMatch ? "match" : "mismatch"}`}>
                    {liveIsMatch ? "UYUMLU" : "UYUMSUZ"}
                  </p>
                  <p>Guven: {Number(liveResult.confidence || 0).toFixed(1)}%</p>
                  <p>Sure: {liveResult.processing_ms ?? "-"} ms</p>
                  <p>Model: {liveResult.ai_provider || "-"}{liveResult.ai_model ? ` / ${liveResult.ai_model}` : ""}</p>
                  {!liveIsMatch && liveResult.analysis_predicted_plu && (
                    <p>
                      En yakin PLU: {liveResult.analysis_predicted_plu}
                      {liveResult.analysis_predicted_score ? ` (${liveResult.analysis_predicted_score})` : ""}
                    </p>
                  )}
                  <p>Top eslesmeler: {summarizeTopMatches(liveResult.top_matches)}</p>
                </div>
                <div className="live-preview">
                  {liveImageBase64 ? (
                    <img
                      src={`data:image/jpeg;base64,${liveImageBase64}`}
                      alt="canli-sonuc"
                      className="live-preview-image"
                    />
                  ) : (
                    <div className="live-preview-empty">Goruntu yok</div>
                  )}
                </div>
              </div>
            ) : (
              <p className="live-placeholder">Canli sonuc bekleniyor...</p>
            )}
          </CardContent>
        </Card>
      )}

      {systemMode === "training" && plusList.length > 0 && (
        <Card className="info-card" data-testid="training-info-card">
          <CardContent className="info-content">
            <div className="info-icon">ℹ️</div>
            <div>
              <h3>Egitim Modu Aktif</h3>
              <p>Sistem su anda veri toplama modunda. Her PLU seciminde fotograf cekiliyor ve kaydediliyor.</p>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default ScaleOperator;
