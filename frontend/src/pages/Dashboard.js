import React, { useState, useEffect } from "react";
import axios from "axios";
import { API } from "@/App";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { toast } from "sonner";
import "./Dashboard.css";

const IMAGE_PAGE_SIZE = 20;
const VALIDATION_PAGE_SIZE = 20;
const REF_CANDIDATE_PAGE_SIZE = 20;

const Dashboard = () => {
  const [stats, setStats] = useState(null);
  const [validationResults, setValidationResults] = useState([]);
  const [capturedImages, setCapturedImages] = useState([]);
  const [loading, setLoading] = useState(true);
  const [systemMode, setSystemMode] = useState("training");
  const [selectedImage, setSelectedImage] = useState(null);
  const [selectedValidation, setSelectedValidation] = useState(null);
  const [selectedRefCandidate, setSelectedRefCandidate] = useState(null);
  const [imageModalOpen, setImageModalOpen] = useState(false);
  const [loadingImage, setLoadingImage] = useState(false);
  const [modalImageMode, setModalImageMode] = useState("processed");
  const [imagesPage, setImagesPage] = useState(1);
  const [imagesLoading, setImagesLoading] = useState(false);
  const [validationsPage, setValidationsPage] = useState(1);
  const [validationsLoading, setValidationsLoading] = useState(false);
  const [aiConfig, setAiConfig] = useState({ provider: "local", model: "", version: "" });
  const [savingAiConfig, setSavingAiConfig] = useState(false);
  const [health, setHealth] = useState(null);
  const [healthLoading, setHealthLoading] = useState(false);
  const [refCandidates, setRefCandidates] = useState([]);
  const [refCandidatesPage, setRefCandidatesPage] = useState(1);
  const [refCandidatesLoading, setRefCandidatesLoading] = useState(false);
  const [refCandidatesStatus, setRefCandidatesStatus] = useState("pending");
  const [activeTab, setActiveTab] = useState("images");

  const formatModel = (provider, model) => {
    const base = provider || "?";
    return model ? `${base} / ${model}` : base;
  };

  const formatDuration = (ms) => {
    if (ms === null || ms === undefined) return "-";
    return `${ms} ms`;
  };

  const formatSimilarityPercent = (rawValue) => {
    if (rawValue === null || rawValue === undefined || rawValue === "") return "-";
    const num = Number(rawValue);
    if (!Number.isFinite(num)) return "-";
    const normalized = num <= 1 ? num * 100 : num;
    return `${normalized.toFixed(1)}%`;
  };

  const formatTopMatchScore = (match) => {
    if (!match) return "-";
    const raw = Number(match.score);
    if (!Number.isFinite(raw)) return "-";
    if (match.score_type === "cosine_similarity") {
      return raw.toFixed(4);
    }
    return `${raw.toFixed(1)}%`;
  };

  const summarizeTopMatches = (matches, limit = 3) => {
    if (!Array.isArray(matches) || matches.length === 0) return "-";
    return matches
      .slice(0, limit)
      .map((match, idx) => {
        const rank = match?.rank ?? idx + 1;
        const pluCode = match?.plu_code || "?";
        return `${rank}. ${pluCode} (${formatTopMatchScore(match)})`;
      })
      .join(" | ");
  };

  const isResnetTopKResult = (result) => result?.source === "resnet_topk";

  const topMatchesLimitFor = (result) => (
    isResnetTopKResult(result) || result?.ai_provider === "butcher_resnet" ? 5 : 3
  );

  useEffect(() => {
    fetchDashboardData();
  }, []);

  useEffect(() => {
    if (activeTab === "images") {
      fetchCapturedImages(imagesPage);
    }
  }, [imagesPage, activeTab]);

  useEffect(() => {
    if (activeTab === "validations") {
      fetchValidationResults(validationsPage);
    }
  }, [validationsPage, activeTab]);

  useEffect(() => {
    if (activeTab === "ref-candidates") {
      fetchRefCandidates(refCandidatesPage, refCandidatesStatus);
    }
  }, [refCandidatesPage, refCandidatesStatus, activeTab]);

  useEffect(() => {
    if (!stats) {
      return;
    }
    const maxPage = Math.max(1, Math.ceil(stats.total_images / IMAGE_PAGE_SIZE));
    if (imagesPage > maxPage) {
      setImagesPage(maxPage);
    }
  }, [stats, imagesPage]);

  useEffect(() => {
    if (!stats) {
      return;
    }
    const maxPage = Math.max(1, Math.ceil(stats.total_validations / VALIDATION_PAGE_SIZE));
    if (validationsPage > maxPage) {
      setValidationsPage(maxPage);
    }
  }, [stats, validationsPage]);

  const fetchDashboardData = async () => {
    try {
      const [statsRes, modeRes, aiRes] = await Promise.all([
        axios.get(`${API}/stats/dashboard`),
        axios.get(`${API}/system/mode`),
        axios.get(`${API}/system/ai-config`)
      ]);

      setStats(statsRes.data);
      setSystemMode(modeRes.data.mode);
      setAiConfig(aiRes.data);
      setLoading(false);
    } catch (error) {
      console.error("Error fetching dashboard data:", error);
      toast.error("Dashboard verileri yuklenemedi");
      setLoading(false);
    }
  };

  const fetchCapturedImages = async (page = 1) => {
    setImagesLoading(true);
    try {
      const skip = (page - 1) * IMAGE_PAGE_SIZE;
      const res = await axios.get(
        `${API}/images/captured?limit=${IMAGE_PAGE_SIZE}&skip=${skip}`
      );
      setCapturedImages(res.data || []);
    } catch (error) {
      console.error("Error fetching captured images:", error);
      toast.error("Fotograf listesi yuklenemedi");
    } finally {
      setImagesLoading(false);
    }
  };

  const fetchValidationResults = async (page = 1) => {
    setValidationsLoading(true);
    try {
      const skip = (page - 1) * VALIDATION_PAGE_SIZE;
      const res = await axios.get(
        `${API}/validation/results?limit=${VALIDATION_PAGE_SIZE}&skip=${skip}`
      );
      setValidationResults(res.data || []);
    } catch (error) {
      console.error("Error fetching validation results:", error);
      toast.error("AI kontrol listesi yuklenemedi");
    } finally {
      setValidationsLoading(false);
    }
  };

  const fetchRefCandidates = async (page = 1, status = "pending") => {
    setRefCandidatesLoading(true);
    try {
      const skip = (page - 1) * REF_CANDIDATE_PAGE_SIZE;
      const res = await axios.get(
        `${API}/ref-candidates?status=${status}&limit=${REF_CANDIDATE_PAGE_SIZE}&skip=${skip}`
      );
      setRefCandidates(res.data || []);
    } catch (error) {
      console.error("Error fetching ref candidates:", error);
      toast.error("Ref aday listesi yuklenemedi");
    } finally {
      setRefCandidatesLoading(false);
    }
  };

  const handleApproveCandidate = async (candidateId) => {
    try {
      await axios.post(`${API}/ref-candidates/${candidateId}/approve`);
      toast.success("Aday onaylandi");
      fetchRefCandidates(refCandidatesPage, refCandidatesStatus);
    } catch (error) {
      console.error("Error approving candidate:", error);
      toast.error("Aday onaylanamadi");
    }
  };

  const handleRejectCandidate = async (candidateId) => {
    try {
      await axios.post(`${API}/ref-candidates/${candidateId}/reject`);
      toast.success("Aday reddedildi");
      fetchRefCandidates(refCandidatesPage, refCandidatesStatus);
    } catch (error) {
      console.error("Error rejecting candidate:", error);
      toast.error("Aday reddedilemedi");
    }
  };

  const fetchHealth = async () => {
    setHealthLoading(true);
    try {
      const res = await axios.get(`${API}/health`);
      setHealth(res.data);
    } catch (error) {
      console.error("Error fetching health:", error);
      toast.error("Sistem durumu alınamadı");
    } finally {
      setHealthLoading(false);
    }
  };

  const handleProviderChange = (provider) => {
    if (
      provider === "local" ||
      provider === "local_large" ||
      provider === "local_embedding" ||
      provider === "butcher_resnet"
    ) {
      setAiConfig({ provider, model: "" });
    } else if (
      provider === "gemini" ||
      provider === "local_gemini" ||
      provider === "local_gemini_consensus"
    ) {
      setAiConfig({ provider, model: aiConfig.model || "gemini-2.5-flash-lite" });
    } else {
      setAiConfig({ provider, model: aiConfig.model || "" });
    }
  };

  const handleModeToggle = async () => {
    const newMode = systemMode === "training" ? "production" : "training";
    try {
      await axios.post(`${API}/system/mode`, { mode: newMode });
      setSystemMode(newMode);
      toast.success(`Sistem modu ${newMode === "training" ? "Egitim" : "Uretim"} moduna gecti`);
      fetchDashboardData();
    } catch (error) {
      console.error("Error toggling mode:", error);
      toast.error("Mod degistirilemedi");
    }
  };

  const handleImageClick = async (imageId) => {
    setLoadingImage(true);
    setImageModalOpen(true);
    setModalImageMode("processed");
    setSelectedValidation(null);
    try {
      const response = await axios.get(`${API}/images/${imageId}`);
      setSelectedImage(response.data);
    } catch (error) {
      console.error("Error fetching image:", error);
      toast.error("Fotograf yuklenemedi");
      setImageModalOpen(false);
    } finally {
      setLoadingImage(false);
    }
  };

  const handleValidationClick = async (validationId) => {
    setLoadingImage(true);
    setImageModalOpen(true);
    setModalImageMode("processed");
    setSelectedImage(null);
    try {
      const response = await axios.get(`${API}/validation/${validationId}`);
      setSelectedValidation(response.data);
    } catch (error) {
      console.error("Error fetching validation image:", error);
      toast.error("Fotograf yuklenemedi");
      setImageModalOpen(false);
    } finally {
      setLoadingImage(false);
    }
  };

  const closeImageModal = () => {
    setImageModalOpen(false);
    setModalImageMode("processed");
    setSelectedImage(null);
    setSelectedValidation(null);
    setSelectedRefCandidate(null);
  };

  const handleRefCandidateClick = (candidate) => {
    setLoadingImage(false);
    setImageModalOpen(true);
    setModalImageMode("processed");
    setSelectedImage(null);
    setSelectedValidation(null);
    setSelectedRefCandidate(candidate);
  };

  const handleAiConfigSave = async () => {
    setSavingAiConfig(true);
    try {
      await axios.post(`${API}/system/ai-config`, aiConfig);
      toast.success("Model ayari guncellendi. Sonraki fotograf bu modelle analiz edilecek.");
    } catch (error) {
      console.error("Error updating AI config:", error);
      toast.error("Model ayari kaydedilemedi");
    } finally {
      setSavingAiConfig(false);
    }
  };

  const totalImages = stats?.total_images ?? 0;
  const imagesTotalPages = Math.max(1, Math.ceil(totalImages / IMAGE_PAGE_SIZE));
  const totalValidations = stats?.total_validations ?? 0;
  const validationsTotalPages = Math.max(1, Math.ceil(totalValidations / VALIDATION_PAGE_SIZE));

  if (loading) {
    return (
      <div className="page-container">
        <div className="spinner"></div>
      </div>
    );
  }

  return (
    <div className="page-container dashboard" data-testid="dashboard-page">
      <div className="dashboard-header">
        <div>
          <h1 data-testid="dashboard-title">Dashboard</h1>
          <p className="subtitle" data-testid="dashboard-subtitle">Sistem istatistikleri ve raporlar</p>
        </div>
        <div className="header-actions">
          <div className={`mode-badge ${systemMode}`} data-testid="mode-badge">
            <span className="mode-indicator"></span>
            {systemMode === "training" ? "Egitim Modu" : "Uretim Modu"}
          </div>
          <Button onClick={handleModeToggle} data-testid="mode-toggle-btn">
            {systemMode === "training" ? "Üretim Moduna Geç" : "Egitim Moduna Geç"}
          </Button>
        </div>
      </div>

      <Card className="ai-config-card" data-testid="ai-config-card">
        <CardHeader>
          <CardTitle>Model Seçimi</CardTitle>
          <CardDescription>Sonraki fotoğraf analizinde kullanılacak model</CardDescription>
        </CardHeader>
        <CardContent className="ai-config-content">
          <div className="ai-config-row">
            <label htmlFor="ai-provider">Sağlayıcı</label>
            <select
              id="ai-provider"
              value={aiConfig.provider}
              onChange={(e) => handleProviderChange(e.target.value)}
              data-testid="ai-provider-select"
            >
              <option value="local">Local (ONNX)</option>
              <option value="local_large">Local (Large ONNX)</option>
              <option value="local_embedding">Local (Embedding)</option>
              <option value="butcher_resnet">Butcher ResNet18 (Birebir)</option>
              <option value="local_gemini">Local + Gemini (fallback)</option>
              <option value="local_gemini_consensus">Local + Gemini (consensus)</option>
              <option value="gemini">Gemini</option>
              <option value="openai">OpenAI</option>
            </select>
          </div>
          {(aiConfig.provider === "gemini" || aiConfig.provider === "local_gemini" || aiConfig.provider === "local_gemini_consensus") && (
            <div className="ai-config-row">
              <label htmlFor="ai-model">Model Adı</label>
              <select
                id="ai-model"
                value={aiConfig.model || "gemini-2.5-flash-lite"}
                onChange={(e) => setAiConfig({ ...aiConfig, model: e.target.value })}
                data-testid="ai-model-select"
              >
                  <option value="gemini-2.5-flash-lite">gemini-2.5-flash-lite</option>
                  <option value="gemini-2.5-flash">gemini-2.5-flash</option>
                  <option value="gemini-3-flash">gemini-3-flash</option>
                  <option value="gemma-3-4b">gemma-3-4b</option>
                  <option value="gemma-3-12b">gemma-3-12b</option>
                  <option value="gemma-3-27b">gemma-3-27b</option>
                </select>
              </div>
            )}
          {aiConfig.provider === "openai" && (
            <div className="ai-config-row">
              <label htmlFor="ai-model">Model Adı</label>
              <input
                id="ai-model"
                type="text"
                value={aiConfig.model || ""}
                onChange={(e) => setAiConfig({ ...aiConfig, model: e.target.value })}
                placeholder="Örn: gpt-5"
                data-testid="ai-model-input"
              />
            </div>
          )}
          <Button onClick={handleAiConfigSave} disabled={savingAiConfig} data-testid="save-ai-config-btn">
            {savingAiConfig ? "Kaydediliyor..." : "Kaydet"}
          </Button>
        </CardContent>
      </Card>

      <div className="stats-grid" data-testid="stats-grid">
        <Card className="stat-card">
          <CardContent>
            <div className="stat-value" data-testid="total-images-stat">{stats.total_images}</div>
            <div className="stat-label">Toplam Fotoğraf</div>
          </CardContent>
        </Card>
        
        <Card className="stat-card">
          <CardContent>
            <div className="stat-value" data-testid="total-validations-stat">{stats.total_validations}</div>
            <div className="stat-label">AI Kontrol Sayısı</div>
          </CardContent>
        </Card>
        
        <Card className="stat-card">
          <CardContent>
            <div className="stat-value match" data-testid="match-count-stat">{stats.match_count}</div>
            <div className="stat-label">Uyumlu</div>
          </CardContent>
        </Card>
        
        <Card className="stat-card">
          <CardContent>
            <div className="stat-value mismatch" data-testid="mismatch-count-stat">{stats.mismatch_count}</div>
            <div className="stat-label">Uyumsuz</div>
          </CardContent>
        </Card>
      </div>

      {(stats.match_count + stats.mismatch_count) > 0 && (
        <Card className="accuracy-card" data-testid="accuracy-card">
          <CardHeader>
            <CardTitle>Doğruluk Oranı</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="accuracy-display">
              <div className="accuracy-circle">
                <div className="accuracy-value" data-testid="accuracy-percentage">{stats.match_percentage}%</div>
              </div>
              <div className="accuracy-details">
                <p>Toplam {stats.match_count + stats.mismatch_count} uyumluluk kontrolu</p>
                <p className="match-text">✔ {stats.match_count} uyumlu tespit</p>
                <p className="mismatch-text">✖ {stats.mismatch_count} uyumsuz tespit</p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      <Tabs
        value={activeTab}
        onValueChange={setActiveTab}
        className="data-tabs"
        data-testid="data-tabs"
      >
        <TabsList>
          <TabsTrigger value="images" data-testid="images-tab">Fotoğraflar ({totalImages || capturedImages.length})</TabsTrigger>
          <TabsTrigger value="validations" data-testid="validations-tab">AI Kontrolleri ({totalValidations || validationResults.length})</TabsTrigger>
          <TabsTrigger value="ref-candidates" data-testid="ref-candidates-tab">Ref Adaylari ({refCandidates.length})</TabsTrigger>
          <TabsTrigger value="plu-stats" data-testid="plu-stats-tab">PLU istatistikleri</TabsTrigger>
          <TabsTrigger value="health" data-testid="health-tab">Sistem Durumu</TabsTrigger>
        </TabsList>
        
        <TabsContent value="images" data-testid="images-tab-content">
          <div className="data-grid">
            {capturedImages.length === 0 ? (
              <Card className="empty-state">
                <CardContent className="empty-content">
                  <div className="empty-icon">📷</div>
                  <h3>Henüz fotoğraf yok</h3>
                  <p>PLU seçimi yaparak fotoğraf çekmeye başlayın</p>
                </CardContent>
              </Card>
            ) : (
              capturedImages.map((img) => (
                <Card 
                  key={img.id} 
                  className="image-card clickable" 
                  data-testid={`image-card-${img.id}`}
                  onClick={() => handleImageClick(img.id)}
                  style={{ cursor: 'pointer' }}
                >
                  <CardHeader>
                    <CardTitle data-testid={`image-plu-${img.id}`}>PLU {img.plu_code}</CardTitle>
                    <CardDescription data-testid={`image-timestamp-${img.id}`}>
                      {new Date(img.timestamp).toLocaleString('tr-TR')}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className={`phase-badge ${img.phase}`} data-testid={`image-phase-${img.id}`}>
                      {img.phase === "training" ? "Egitim" : "Uretim"}
                    </div>
                    <p className="model-chip" data-testid={`image-model-${img.id}`}>
                      Model: {formatModel(img.ai_provider, img.ai_model)}
                    </p>
                    <p className="click-hint">Fotoğrafı görmek için tıklayın</p>
                  </CardContent>
                </Card>
              ))
            )}
          </div>
          {capturedImages.length > 0 && (
            <div className="data-pagination">
              <Button
                variant="outline"
                onClick={() => setImagesPage((page) => Math.max(1, page - 1))}
                disabled={imagesPage <= 1 || imagesLoading}
              >
                Onceki
              </Button>
              <span className="page-info">
                Sayfa {imagesPage} / {imagesTotalPages}
              </span>
              <Button
                variant="outline"
                onClick={() => setImagesPage((page) => Math.min(imagesTotalPages, page + 1))}
                disabled={imagesPage >= imagesTotalPages || imagesLoading}
              >
                Sonraki
              </Button>
            </div>
          )}
        </TabsContent>
        
        <TabsContent value="validations" data-testid="validations-tab-content">
          <div className="data-grid">
            {validationResults.length === 0 ? (
              <Card className="empty-state">
                <CardContent className="empty-content">
                  <div className="empty-icon">🤖</div>
                  <h3>Henüz AI kontrolü yapılmadı</h3>
                  <p>Uretim moduna geçerek AI kontrollerini başlatın</p>
                </CardContent>
              </Card>
            ) : (
              validationResults.map((result) => {
                const isTopKOnly = isResnetTopKResult(result);
                return (
                  <Card
                    key={result.id}
                    className={`validation-card ${isTopKOnly ? "topk" : (result.is_match ? "match" : "mismatch")} clickable`}
                    data-testid={`validation-card-${result.id}`}
                    onClick={() => handleValidationClick(result.id)}
                    style={{ cursor: "pointer" }}
                  >
                    <CardHeader>
                      <CardTitle data-testid={`validation-plu-${result.id}`}>
                        <span>{result.selected_plu_name}</span>
                        {isTopKOnly ? (
                          <span className="result-badge neutral" data-testid={`validation-result-${result.id}`}>
                            Top-5
                          </span>
                        ) : (
                          <span className={`result-badge ${result.is_match ? "match" : "mismatch"}`} data-testid={`validation-result-${result.id}`}>
                            {result.is_match ? "✔ Uyumlu" : "✖ Uyumsuz"}
                          </span>
                        )}
                      </CardTitle>
                      <CardDescription data-testid={`validation-timestamp-${result.id}`}>
                        {new Date(result.timestamp).toLocaleString('tr-TR')}
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <p className="model-chip" data-testid={`validation-model-${result.id}`}>
                        Model: {formatModel(result.ai_provider, result.ai_model)}
                      </p>
                      <p className="model-chip" data-testid={`validation-duration-${result.id}`}>
                        Süre: {result.processing_ms !== undefined && result.processing_ms !== null ? `${result.processing_ms} ms` : "-"}
                      </p>
                      {!isTopKOnly && !result.is_match && result.analysis_predicted_plu && (
                        <p className="model-chip" data-testid={`validation-nearest-plu-${result.id}`}>
                          En yakin PLU: {result.analysis_predicted_plu} ({formatSimilarityPercent(result.analysis_predicted_score)})
                        </p>
                      )}
                      {Array.isArray(result.top_matches) && result.top_matches.length > 0 && (
                        <p className="model-chip" data-testid={`validation-top-matches-${result.id}`}>
                          Top {topMatchesLimitFor(result)} benzer: {summarizeTopMatches(result.top_matches, topMatchesLimitFor(result))}
                        </p>
                      )}
                      {result.has_processed_image && (
                        <p className="model-chip" data-testid={`validation-roi-${result.id}`}>
                          ROI: crop kaydi var
                        </p>
                      )}
                      {(result.ai_provider === "local_gemini" || result.ai_provider === "local_gemini_consensus") && (
                        <div className="fallback-details">
                          <p className="model-chip" data-testid={`validation-local-${result.id}`}>
                            Local: {result.fallback_local_match === true ? "Uyumlu" : result.fallback_local_match === false ? "Uyumsuz" : "-"} ({result.fallback_local_confidence ?? "-"}%)
                          </p>
                          <p className="model-chip" data-testid={`validation-remote-${result.id}`}>
                            Gemini: {result.fallback_remote_match === true ? "Uyumlu" : result.fallback_remote_match === false ? "Uyumsuz" : "-"} ({result.fallback_remote_confidence ?? "-"}%)
                          </p>
                        </div>
                      )}

                      <div className="confidence-bar">
                        <div className="confidence-label" data-testid={`validation-confidence-${result.id}`}>{isTopKOnly ? "Top-1 Skor" : "Güven"}: {result.confidence}%</div>
                        <div className="confidence-progress">
                          <div
                            className="confidence-fill"
                            style={{width: `${result.confidence}%`}}
                          ></div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                );
              })
            )}
          </div>
          {validationResults.length > 0 && (
            <div className="data-pagination">
              <Button
                variant="outline"
                onClick={() => setValidationsPage((page) => Math.max(1, page - 1))}
                disabled={validationsPage <= 1 || validationsLoading}
              >
                Onceki
              </Button>
              <span className="page-info">
                Sayfa {validationsPage} / {validationsTotalPages}
              </span>
              <Button
                variant="outline"
                onClick={() => setValidationsPage((page) => Math.min(validationsTotalPages, page + 1))}
                disabled={validationsPage >= validationsTotalPages || validationsLoading}
              >
                Sonraki
              </Button>
            </div>
          )}
        </TabsContent>

        <TabsContent value="ref-candidates" data-testid="ref-candidates-tab-content">
          <div className="ref-candidates-toolbar">
            <span className="ref-status-label">Durum:</span>
            <button
              type="button"
              className={`ref-status-chip ${refCandidatesStatus === "pending" ? "active" : ""}`}
              onClick={() => {
                setRefCandidatesStatus("pending");
                setRefCandidatesPage(1);
              }}
            >
              Bekleyen
            </button>
            <button
              type="button"
              className={`ref-status-chip ${refCandidatesStatus === "approved" ? "active" : ""}`}
              onClick={() => {
                setRefCandidatesStatus("approved");
                setRefCandidatesPage(1);
              }}
            >
              Onaylanan
            </button>
            <button
              type="button"
              className={`ref-status-chip ${refCandidatesStatus === "rejected" ? "active" : ""}`}
              onClick={() => {
                setRefCandidatesStatus("rejected");
                setRefCandidatesPage(1);
              }}
            >
              Reddedilen
            </button>
          </div>
          <div className="data-grid ref-candidates-grid">
            {refCandidates.length === 0 ? (
              <Card className="empty-state">
                <CardContent className="empty-content">
                  <div className="empty-icon">🗂️</div>
                  <h3>Aday yok</h3>
                  <p>Bu durum icin kayit bulunamadi</p>
                </CardContent>
              </Card>
            ) : (
              refCandidates.map((cand) => (
                <Card key={cand.id} className="candidate-card">
                  <CardHeader>
                    <CardTitle>PLU {cand.plu_code}</CardTitle>
                    <CardDescription>
                      {cand.created_at ? new Date(cand.created_at).toLocaleString("tr-TR") : "-"}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="candidate-image-wrap">
                      {cand.image_base64 ? (
                        <img
                          src={`data:image/jpeg;base64,${cand.image_base64}`}
                          alt={`ref-${cand.plu_code}`}
                          className="candidate-image"
                          onClick={() => handleRefCandidateClick(cand)}
                        />
                      ) : (
                        <div className="candidate-image-fallback">Image yok</div>
                      )}
                    </div>
                    <div className="candidate-actions">
                      {(refCandidatesStatus === "pending" || refCandidatesStatus === "rejected") && (
                        <>
                          <Button
                            variant="outline"
                            onClick={() => handleApproveCandidate(cand.id)}
                            disabled={refCandidatesLoading}
                          >
                            Approve
                          </Button>
                          <Button
                            variant="outline"
                            onClick={() => handleRejectCandidate(cand.id)}
                            disabled={refCandidatesLoading}
                          >
                            Reject
                          </Button>
                        </>
                      )}
                    </div>
                  </CardContent>
                </Card>
              ))
            )}
          </div>
          {refCandidates.length > 0 && (
            <div className="data-pagination">
              <Button
                variant="outline"
                onClick={() => setRefCandidatesPage((page) => Math.max(1, page - 1))}
                disabled={refCandidatesPage <= 1 || refCandidatesLoading}
              >
                Onceki
              </Button>
              <span className="page-info">
                Sayfa {refCandidatesPage}
              </span>
              <Button
                variant="outline"
                onClick={() => setRefCandidatesPage((page) => page + 1)}
                disabled={refCandidatesLoading}
              >
                Sonraki
              </Button>
            </div>
          )}
        </TabsContent>
        
        <TabsContent value="plu-stats" data-testid="plu-stats-tab-content">
          <div className="plu-stats-grid">
            {Object.keys(stats.images_by_plu).length === 0 ? (
              <Card className="empty-state">
                <CardContent className="empty-content">
                  <div className="empty-icon">📊</div>
                  <h3>Henüz veri yok</h3>
                </CardContent>
              </Card>
            ) : (
              Object.entries(stats.images_by_plu).map(([plu_code, count]) => (
                <Card key={plu_code} className="plu-stat-card" data-testid={`plu-stat-${plu_code}`}>
                  <CardHeader>
                    <CardTitle data-testid={`plu-stat-code-${plu_code}`}>PLU {plu_code}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="plu-stat-value" data-testid={`plu-stat-count-${plu_code}`}>{count}</div>
                    <div className="plu-stat-label">Fotoğraf</div>
                  </CardContent>
                </Card>
              ))
            )}
          </div>
        </TabsContent>

        <TabsContent value="health" data-testid="health-tab-content">
          <Card className="health-card">
            <CardHeader>
              <CardTitle>Sistem Durumu</CardTitle>
              <CardDescription>Kamera, veritabanı ve kaynak özetleri</CardDescription>
            </CardHeader>
            <CardContent className="health-grid">
              <div className="health-item">
                <span>MongoDB</span>
                <span className={health?.mongo_connected ? "ok" : "warn"}>
                  {health?.mongo_connected ? "Bağlı" : "Sorun"}
                </span>
              </div>
              <div className="health-item">
                <span>Kamera</span>
                <span className={health?.camera_available ? "ok" : "warn"}>
                  {health?.camera_available ? "Hazır" : "Yok"}
                </span>
              </div>
              <div className="health-item">
                <span>AI Modeli</span>
                <span>{health ? `${health.ai_provider}${health.ai_model ? " / " + health.ai_model : ""}` : "-"}</span>
              </div>
              <div className="health-item">
                <span>CPU</span>
                <span>{health ? `${health.cpu_percent}%` : "-"}</span>
              </div>
              <div className="health-item">
                <span>RAM</span>
                <span>{health ? `${health.mem_percent}%` : "-"}</span>
              </div>
              <div className="health-item">
                <span>Disk</span>
                <span>{health ? `${health.disk_percent}%` : "-"}</span>
              </div>
              <Button onClick={fetchHealth} disabled={healthLoading} style={{ gridColumn: "1 / -1" }}>
                {healthLoading ? "Güncelleniyor..." : "Durumu Yenile"}
              </Button>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Image Modal */}
      {imageModalOpen && (
        <div className="image-modal-overlay" onClick={closeImageModal}>
          <div className="image-modal" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={closeImageModal} data-testid="close-image-modal">
              ✕
            </button>
            
            {loadingImage ? (
              <div className="modal-loading">
                <div className="spinner"></div>
                <p>Fotograf yukleniyor...</p>
              </div>
            ) : (selectedImage || selectedValidation || selectedRefCandidate) ? (
              <div className="modal-content">
                {(() => {
                  const modalData = selectedImage || selectedValidation || selectedRefCandidate;
                  const title = modalData?.selected_plu_name
                    ? `${modalData.selected_plu_name} (PLU ${modalData.plu_code})`
                    : `PLU ${modalData?.plu_code ?? "-"}`;
                  const rawTime = modalData?.timestamp || modalData?.created_at;
                  const timestamp = rawTime ? new Date(rawTime).toLocaleString("tr-TR") : "-";
                  const phase = modalData?.phase;
                  const hasMatch = typeof modalData?.is_match === "boolean";
                  const isTopKOnly = isResnetTopKResult(modalData);
                  const hasProcessedImage = Boolean(modalData?.processed_image_base64);
                  const hasOriginalImage = Boolean(modalData?.image_base64);
                  const canSwitchImage = hasProcessedImage && hasOriginalImage;
                  const selectedBase64 =
                    modalImageMode === "processed"
                      ? (modalData?.processed_image_base64 || modalData?.image_base64 || null)
                      : (modalData?.image_base64 || modalData?.processed_image_base64 || null);
                  return (
                    <>
                      <div className="modal-header">
                        <h2>{title}</h2>
                        <p>{timestamp}</p>
                        {phase && (
                          <span className={`phase-badge ${phase}`}>
                            {phase === "training" ? "Egitim" : "Uretim"}
                          </span>
                        )}
                        {hasMatch && !isTopKOnly && (
                          <span className={`result-badge ${modalData.is_match ? "match" : "mismatch"}`}>
                            {modalData.is_match ? "Match" : "Mismatch"}
                          </span>
                        )}
                        {isTopKOnly && (
                          <span className="result-badge neutral">
                            Top-5
                          </span>
                        )}
                      </div>

                      <div className="modal-image-container">
                        {canSwitchImage && (
                          <div className="modal-image-switch">
                            <button
                              type="button"
                              className={`modal-image-switch-btn ${modalImageMode === "processed" ? "active" : ""}`}
                              onClick={() => setModalImageMode("processed")}
                            >
                              Crop
                            </button>
                            <button
                              type="button"
                              className={`modal-image-switch-btn ${modalImageMode === "original" ? "active" : ""}`}
                              onClick={() => setModalImageMode("original")}
                            >
                              Orijinal
                            </button>
                          </div>
                        )}
                        {selectedBase64 ? (
                          <img
                            src={`data:image/jpeg;base64,${selectedBase64}`}
                            alt={title}
                            className="modal-image"
                          />
                        ) : (
                          <p>Image not available</p>
                        )}
                      </div>

                      <div className="modal-info">
                        <p><strong>ID:</strong> {modalData?.id || "-"}</p>
                        <p><strong>Zaman:</strong> {timestamp}</p>
                        {!isTopKOnly && !modalData?.is_match && modalData?.analysis_predicted_plu && (
                          <p>
                            <strong>En yakin PLU:</strong> {modalData.analysis_predicted_plu} ({formatSimilarityPercent(modalData.analysis_predicted_score)})
                          </p>
                        )}
                        {Array.isArray(modalData?.top_matches) && modalData.top_matches.length > 0 && (
                          <p>
                            <strong>Top {topMatchesLimitFor(modalData)} benzer:</strong> {summarizeTopMatches(modalData.top_matches, topMatchesLimitFor(modalData))}
                          </p>
                        )}
                        <p>
                          <strong>Goruntu:</strong>{" "}
                          {modalImageMode === "processed"
                            ? (hasProcessedImage ? "Crop" : "Orijinal")
                            : (hasOriginalImage ? "Orijinal" : "Crop")}
                        </p>
                      </div>
                    </>
                  );
                })()}
              </div>
            ) : (
              <div className="modal-error">
                <p>Fotograf yuklenemedi</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default Dashboard;
