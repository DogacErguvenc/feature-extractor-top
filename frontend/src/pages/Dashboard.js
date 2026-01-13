import React, { useState, useEffect } from "react";
import axios from "axios";
import { API } from "@/App";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { toast } from "sonner";
import "./Dashboard.css";

const IMAGE_PAGE_SIZE = 20;

const Dashboard = () => {
  const [stats, setStats] = useState(null);
  const [validationResults, setValidationResults] = useState([]);
  const [capturedImages, setCapturedImages] = useState([]);
  const [loading, setLoading] = useState(true);
  const [systemMode, setSystemMode] = useState("training");
  const [selectedImage, setSelectedImage] = useState(null);
  const [imageModalOpen, setImageModalOpen] = useState(false);
  const [loadingImage, setLoadingImage] = useState(false);
  const [imagesPage, setImagesPage] = useState(1);
  const [imagesLoading, setImagesLoading] = useState(false);
  const [aiConfig, setAiConfig] = useState({ provider: "local", model: "", version: "" });
  const [savingAiConfig, setSavingAiConfig] = useState(false);
  const [health, setHealth] = useState(null);
  const [healthLoading, setHealthLoading] = useState(false);

  const formatModel = (provider, model) => {
    const base = provider || "?";
    return model ? `${base} / ${model}` : base;
  };

  const formatDuration = (ms) => {
    if (ms === null || ms === undefined) return "-";
    return `${ms} ms`;
  };

  useEffect(() => {
    fetchDashboardData();
  }, []);

  useEffect(() => {
    fetchCapturedImages(imagesPage);
  }, [imagesPage]);

  useEffect(() => {
    if (!stats) {
      return;
    }
    const maxPage = Math.max(1, Math.ceil(stats.total_images / IMAGE_PAGE_SIZE));
    if (imagesPage > maxPage) {
      setImagesPage(maxPage);
    }
  }, [stats, imagesPage]);

  const fetchDashboardData = async () => {
    try {
      const [statsRes, validationRes, modeRes, aiRes] = await Promise.all([
        axios.get(`${API}/stats/dashboard`),
        axios.get(`${API}/validation/results?limit=20`),
        axios.get(`${API}/system/mode`),
        axios.get(`${API}/system/ai-config`)
      ]);

      setStats(statsRes.data);
      setValidationResults(validationRes.data);
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
    if (provider === "local" || provider === "local_large" || provider === "local_embedding") {
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

  const closeImageModal = () => {
    setImageModalOpen(false);
    setSelectedImage(null);
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

      {stats.total_validations > 0 && (
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
                <p>Toplam {stats.total_validations} kontrol yapıldı</p>
                <p className="match-text">✔ {stats.match_count} uyumlu tespit</p>
                <p className="mismatch-text">✖ {stats.mismatch_count} uyumsuz tespit</p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      <Tabs defaultValue="images" className="data-tabs" data-testid="data-tabs">
        <TabsList>
          <TabsTrigger value="images" data-testid="images-tab">Fotoğraflar ({totalImages || capturedImages.length})</TabsTrigger>
          <TabsTrigger value="validations" data-testid="validations-tab">AI Kontrolleri ({validationResults.length})</TabsTrigger>
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
            <div className="images-pagination">
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
              validationResults.map((result) => (
                <Card key={result.id} className={`validation-card ${result.is_match ? 'match' : 'mismatch'}`} data-testid={`validation-card-${result.id}`}>
                  <CardHeader>
                    <CardTitle data-testid={`validation-plu-${result.id}`}>
                      <span>{result.selected_plu_name}</span>
                      <span className={`result-badge ${result.is_match ? 'match' : 'mismatch'}`} data-testid={`validation-result-${result.id}`}>
                        {result.is_match ? "✔ Uyumlu" : "✖ Uyumsuz"}
                      </span>
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
                      <div className="confidence-label" data-testid={`validation-confidence-${result.id}`}>Güven: {result.confidence}%</div>
                      <div className="confidence-progress">
                        <div 
                          className="confidence-fill" 
                          style={{width: `${result.confidence}%`}}
                        ></div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))
            )}
          </div>
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
                <p>Fotoğraf yükleniyor...</p>
              </div>
            ) : selectedImage ? (
              <div className="modal-content">
                <div className="modal-header">
                  <h2>PLU {selectedImage.plu_code}</h2>
                  <p>{new Date(selectedImage.timestamp).toLocaleString('tr-TR')}</p>
                  <span className={`phase-badge ${selectedImage.phase}`}>
                    {selectedImage.phase === "training" ? "Egitim" : "Uretim"}
                  </span>
                </div>
                
                <div className="modal-image-container">
                  <img 
                    src={`data:image/jpeg;base64,${selectedImage.image_base64}`}
                    alt={`PLU ${selectedImage.plu_code}`}
                    className="modal-image"
                  />
                </div>
                
                <div className="modal-info">
                  <p><strong>Fotoğraf ID:</strong> {selectedImage.id}</p>
                  <p><strong>Zaman:</strong> {new Date(selectedImage.timestamp).toLocaleString('tr-TR')}</p>
                </div>
              </div>
            ) : (
              <div className="modal-error">
                <p>Fotoğraf yüklenemedi</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default Dashboard;

