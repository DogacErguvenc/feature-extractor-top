import React, { useState, useEffect, useCallback, useRef } from "react";
import axios from "axios";
import { API } from "@/App";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { toast } from "sonner";
import "./ScaleOperator.css";

const LIVE_SUPPORTED_PROVIDERS = new Set([
  "local",
  "local_large",
  "local_embedding",
  "butcher_resnet",
  "butcher_resnet_embedding",
]);

const sleep = (ms) => new Promise((resolve) => window.setTimeout(resolve, ms));

const formatTopMatchScore = (match) => {
  const raw = Number(match?.score);
  if (!Number.isFinite(raw)) return "-";
  if (match?.score_type === "cosine_similarity") {
    return raw.toFixed(4);
  }
  return `${raw.toFixed(1)}%`;
};

const captureVideoFrameBase64 = (videoEl, canvasEl) => {
  if (!videoEl || !canvasEl) return "";
  const width = videoEl.videoWidth;
  const height = videoEl.videoHeight;
  if (!width || !height) return "";

  canvasEl.width = width;
  canvasEl.height = height;
  const ctx = canvasEl.getContext("2d");
  if (!ctx) return "";
  ctx.drawImage(videoEl, 0, 0, width, height);

  const dataUrl = canvasEl.toDataURL("image/jpeg", 0.82);
  const split = dataUrl.split(",");
  if (split.length !== 2) return "";
  return split[1];
};

const ScaleOperator = () => {
  const [plusList, setPlusList] = useState([]);
  const [loading, setLoading] = useState(true);
  const [systemMode, setSystemMode] = useState("training");
  const [processing, setProcessing] = useState(false);
  const [selectedPLU, setSelectedPLU] = useState(null);

  const [aiProvider, setAiProvider] = useState("");
  const [aiModel, setAiModel] = useState("");

  const [liveOpen, setLiveOpen] = useState(false);
  const [liveLoading, setLiveLoading] = useState(false);
  const [liveError, setLiveError] = useState("");
  const [liveTopMatches, setLiveTopMatches] = useState([]);
  const [liveProcessingMs, setLiveProcessingMs] = useState(null);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const liveRequestRef = useRef(false);

  const fetchPLUList = useCallback(async () => {
    try {
      const response = await axios.get(`${API}/plu/list`);
      setPlusList(response.data);
      setLoading(false);
    } catch (error) {
      console.error("Error fetching PLU list:", error);
      toast.error("PLU listesi yuklenemedi");
      setLoading(false);
    }
  }, []);

  const fetchSystemMode = useCallback(async () => {
    try {
      const response = await axios.get(`${API}/system/mode`);
      setSystemMode(response.data.mode);
    } catch (error) {
      console.error("Error fetching system mode:", error);
    }
  }, []);

  const fetchAIConfig = useCallback(async () => {
    try {
      const response = await axios.get(`${API}/system/ai-config`);
      const provider = response.data?.provider || "";
      const model = response.data?.model || "";
      setAiProvider(provider);
      setAiModel(model);
      return { provider, model };
    } catch (error) {
      console.error("Error fetching AI config:", error);
      return { provider: "", model: "" };
    }
  }, []);

  useEffect(() => {
    fetchPLUList();
    fetchSystemMode();
    fetchAIConfig();
  }, [fetchPLUList, fetchSystemMode, fetchAIConfig]);

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

      window.setTimeout(() => {
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

  const stopMediaStream = useCallback(() => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, []);

  const stopLivePredict = useCallback(() => {
    setLiveOpen(false);
    setLiveLoading(false);
    liveRequestRef.current = false;
    stopMediaStream();
  }, [stopMediaStream]);

  const startLivePredict = useCallback(async () => {
    const cfg = await fetchAIConfig();
    if (!LIVE_SUPPORTED_PROVIDERS.has(cfg.provider)) {
      toast.error(
        `Canli tahmin icin model local/local_large/local_embedding/butcher_resnet/butcher_resnet_embedding olmali. Su an: ${cfg.provider || "belirsiz"}`
      );
      return;
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      toast.error("Tarayici kamera erisimini desteklemiyor");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" } },
        audio: false,
      });
      mediaStreamRef.current = stream;
      setLiveError("");
      setLiveTopMatches([]);
      setLiveProcessingMs(null);
      setLiveOpen(true);
    } catch (error) {
      console.error("Error opening browser camera:", error);
      toast.error("Kamera acilamadi. Tarayici izinlerini kontrol et.");
    }
  }, [fetchAIConfig]);

  useEffect(() => {
    if (!liveOpen) return undefined;
    const videoEl = videoRef.current;
    if (!videoEl || !mediaStreamRef.current) return undefined;
    videoEl.srcObject = mediaStreamRef.current;
    const playPromise = videoEl.play();
    if (playPromise && typeof playPromise.catch === "function") {
      playPromise.catch((error) => {
        console.error("Error playing live video:", error);
      });
    }
    return undefined;
  }, [liveOpen]);

  useEffect(() => {
    if (!liveOpen) return undefined;

    let cancelled = false;

    const liveLoop = async () => {
      while (!cancelled) {
        if (liveRequestRef.current) {
          await sleep(25);
          continue;
        }

        const videoEl = videoRef.current;
        if (!videoEl || videoEl.readyState < 2) {
          await sleep(80);
          continue;
        }

        const frameBase64 = captureVideoFrameBase64(videoEl, canvasRef.current);
        if (!frameBase64) {
          await sleep(80);
          continue;
        }

        liveRequestRef.current = true;
        setLiveLoading(true);
        try {
          const response = await axios.post(`${API}/live/predict`, {
            image_base64: frameBase64,
            camera_index: 0,
          });
          const matches = Array.isArray(response.data?.top_matches)
            ? response.data.top_matches.slice(0, 3)
            : [];
          if (!cancelled) {
            setLiveTopMatches(matches);
            setLiveProcessingMs(
              Number.isFinite(Number(response.data?.processing_ms))
                ? Number(response.data.processing_ms)
                : null
            );
            setLiveError("");
          }
        } catch (error) {
          if (!cancelled) {
            console.error("Error getting live prediction:", error);
            setLiveError(error.response?.data?.detail || "Canli tahmin alinamadi");
          }
          await sleep(220);
        } finally {
          liveRequestRef.current = false;
          setLiveLoading(false);
        }

        await sleep(90);
      }
    };

    liveLoop();
    return () => {
      cancelled = true;
      liveRequestRef.current = false;
    };
  }, [liveOpen]);

  useEffect(() => {
    return () => {
      liveRequestRef.current = false;
      stopMediaStream();
    };
  }, [stopMediaStream]);

  if (loading) {
    return (
      <div className="page-container">
        <div className="spinner"></div>
      </div>
    );
  }

  return (
    <div className="page-container scale-operator" data-testid="scale-operator-page">
      <div className="operator-header">
        <div>
          <h1 data-testid="page-title">Terazi Operator Ekrani</h1>
          <p className="subtitle" data-testid="page-subtitle">Lutfen urun icin PLU secimi yapin</p>
        </div>
        <div className="operator-actions">
          <Button
            className={`live-global-btn ${liveOpen ? "stop" : ""}`}
            onClick={liveOpen ? stopLivePredict : startLivePredict}
            disabled={processing}
            data-testid="global-live-toggle-btn"
          >
            {liveOpen ? "Canli Tahmini Durdur" : "Canli Tahmin"}
          </Button>
          <div className={`mode-badge ${systemMode}`} data-testid="system-mode-badge">
            <span className="mode-indicator"></span>
            {systemMode === "training" ? "Egitim Modu" : "Uretim Modu"}
          </div>
        </div>
      </div>

      {liveOpen && (
        <Card className="live-camera-card" data-testid="live-camera-card">
          <CardHeader>
            <CardTitle>Canli Tahmin</CardTitle>
            <CardDescription>
              Model: {aiProvider || "-"}{aiModel ? ` / ${aiModel}` : ""} | Sol ustte en benzer 3 urun skoru
            </CardDescription>
          </CardHeader>
          <CardContent className="live-camera-content">
            {liveError && <p className="live-error">{liveError}</p>}
            <div className="live-camera-stage">
              <video ref={videoRef} className="live-camera-video" autoPlay playsInline muted />
              <div className="live-overlay" data-testid="live-overlay">
                <p className="live-overlay-title">En benzer 3 urun</p>
                {liveTopMatches.length > 0 ? (
                  <ul className="live-overlay-list">
                    {liveTopMatches.map((match, idx) => (
                      <li key={`${match?.plu_code || "?"}-${idx}`}>
                        <span>{idx + 1}. {match?.plu_code || "?"}</span>
                        <span>{formatTopMatchScore(match)}</span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="live-overlay-empty">
                    {liveLoading ? "Tahmin aliniyor..." : "Kamera acildi, sonuc bekleniyor..."}
                  </p>
                )}
                <p className="live-overlay-meta">
                  Sure: {liveProcessingMs != null ? `${liveProcessingMs.toFixed(0)} ms` : "-"}
                </p>
              </div>
            </div>
            <canvas ref={canvasRef} className="live-canvas-hidden" />
          </CardContent>
        </Card>
      )}

      {plusList.length === 0 ? (
        <Card className="empty-state" data-testid="empty-plu-state">
          <CardContent className="empty-content">
            <div className="empty-icon">No PLU</div>
            <h3>Henuz PLU eklenmemis</h3>
            <p>Baslamak icin PLU Yonetimi sayfasindan urun ekleyin</p>
          </CardContent>
        </Card>
      ) : (
        <div className="plu-grid" data-testid="plu-grid">
          {plusList.map((plu) => (
            <Card
              key={plu.id}
              className={`plu-card ${selectedPLU?.id === plu.id ? "selected" : ""} ${processing && selectedPLU?.id === plu.id ? "processing" : ""}`}
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
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {systemMode === "training" && plusList.length > 0 && (
        <Card className="info-card" data-testid="training-info-card">
          <CardContent className="info-content">
            <div className="info-icon">i</div>
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
