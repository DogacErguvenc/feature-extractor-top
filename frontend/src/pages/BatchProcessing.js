import React, { useEffect, useState } from "react";
import axios from "axios";
import { API } from "@/App";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { toast } from "sonner";
import "./BatchProcessing.css";

const detectPluFromName = (name = "") => {
  const match = name.match(/(\d{2,6})/);
  return match ? match[1] : "";
};

const formatTopMatchScore = (match) => {
  const raw = Number(match?.score);
  if (!Number.isFinite(raw)) return "-";
  if (match?.score_type === "cosine_similarity") {
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
      return `${rank}. ${match?.plu_code || "?"} (${formatTopMatchScore(match)})`;
    })
    .join(" | ");
};

const topMatchesLimitFor = (result) => (result?.ai_provider === "butcher_resnet" ? 5 : 3);

const formatSize = (size) => {
  if (!size) return "-";
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${Math.round(size / 1024)} KB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
};

const statusCopy = {
  pending: "Bekliyor",
  processing: "İşleniyor",
  match: "Uyumlu",
  mismatch: "Uyumsuz",
  error: "Hata"
};

const centroidStatusCopy = {
  queued: "Sırada",
  running: "Çalışıyor",
  cancelling: "İptal ediliyor",
  cancelled: "İptal edildi",
  complete: "Tamamlandı",
  error: "Hata"
};

const activeCentroidStatuses = ["queued", "running", "cancelling"];
const centroidStorageKeys = {
  inputDir: "batchProcessing.centroid.inputDir",
  outDir: "batchProcessing.centroid.outDir"
};

const readStoredValue = (key) => {
  if (typeof window === "undefined" || !window.localStorage) {
    return "";
  }
  try {
    return window.localStorage.getItem(key) || "";
  } catch (error) {
    console.warn("Stored setting could not be read", error);
    return "";
  }
};

const writeStoredValue = (key, value) => {
  if (typeof window === "undefined" || !window.localStorage) {
    return;
  }
  try {
    const cleanValue = String(value || "").trim();
    if (cleanValue) {
      window.localStorage.setItem(key, cleanValue);
    } else {
      window.localStorage.removeItem(key);
    }
  } catch (error) {
    console.warn("Stored setting could not be saved", error);
  }
};

const BatchProcessing = () => {
  const [pluList, setPluList] = useState([]);
  const [files, setFiles] = useState([]);
  const [defaultPlu, setDefaultPlu] = useState("");
  const [processing, setProcessing] = useState(false);
  const [summary, setSummary] = useState(null);
  const [folderingEnabled, setFolderingEnabled] = useState(false);
  const [folderingInfo, setFolderingInfo] = useState(null);
  const [centroidInputDir, setCentroidInputDir] = useState(() => readStoredValue(centroidStorageKeys.inputDir));
  const [centroidOutDir, setCentroidOutDir] = useState(() => readStoredValue(centroidStorageKeys.outDir));
  const [centroidRecursive, setCentroidRecursive] = useState(false);
  const [centroidCopyMode, setCentroidCopyMode] = useState("bands");
  const [centroidJob, setCentroidJob] = useState(null);
  const [centroidStarting, setCentroidStarting] = useState(false);
  const [folderPicker, setFolderPicker] = useState({
    open: false,
    target: "input",
    title: "",
    currentPath: "",
    parentPath: null,
    entries: [],
    loading: false,
    error: ""
  });
  const centroidBusy = centroidStarting || activeCentroidStatuses.includes(centroidJob?.status);
  const centroidCanCancel = ["queued", "running"].includes(centroidJob?.status);

  useEffect(() => {
    fetchPluList();
  }, []);

  useEffect(() => {
    writeStoredValue(centroidStorageKeys.inputDir, centroidInputDir);
  }, [centroidInputDir]);

  useEffect(() => {
    writeStoredValue(centroidStorageKeys.outDir, centroidOutDir);
  }, [centroidOutDir]);

  useEffect(() => {
    if (!centroidJob?.job_id || !activeCentroidStatuses.includes(centroidJob.status)) {
      return undefined;
    }
    const timer = window.setInterval(() => {
      fetchCentroidJob(centroidJob.job_id);
    }, 2000);
    return () => window.clearInterval(timer);
  }, [centroidJob?.job_id, centroidJob?.status]);

  const fetchPluList = async () => {
    try {
      const res = await axios.get(`${API}/plu/list`);
      setPluList(res.data || []);
    } catch (error) {
      console.error("Error fetching PLU list", error);
      toast.error("PLU listesi alınamadı");
    }
  };

  const fetchCentroidJob = async (jobId) => {
    try {
      const res = await axios.get(`${API}/dataset/centroid-rank/${jobId}`);
      setCentroidJob(res.data);
      if (res.data?.status === "complete") {
        toast.success("Centroid temizleme tamamlandı");
      } else if (res.data?.status === "cancelled") {
        toast.info("Centroid temizleme iptal edildi");
      } else if (res.data?.status === "error") {
        toast.error(`Centroid temizleme hatası: ${res.data?.error || "Bilinmeyen hata"}`);
      }
    } catch (error) {
      console.error("Error fetching centroid job", error);
      toast.error("Centroid iş durumu alınamadı");
    }
  };

  const fetchFolderList = async (path = "") => {
    setFolderPicker((prev) => ({ ...prev, loading: true, error: "" }));
    try {
      const res = await axios.get(`${API}/dataset/folders`, {
        params: path ? { path } : {}
      });
      setFolderPicker((prev) => ({
        ...prev,
        currentPath: res.data?.current_path || "",
        parentPath: res.data?.parent_path || null,
        entries: res.data?.entries || [],
        loading: false,
        error: ""
      }));
    } catch (error) {
      console.error("Error fetching folders", error);
      const message = error.response?.data?.detail || "Klasör listesi alınamadı";
      setFolderPicker((prev) => ({
        ...prev,
        loading: false,
        error: message
      }));
      toast.error(message);
    }
  };

  const openFolderPicker = (target) => {
    const selectedPath = target === "input" ? centroidInputDir : centroidOutDir;
    setFolderPicker({
      open: true,
      target,
      title: target === "input" ? "Karışık fotoğraf klasörü seç" : "Çıktı klasörü seç",
      currentPath: "",
      parentPath: null,
      entries: [],
      loading: true,
      error: ""
    });
    fetchFolderList(selectedPath || "");
  };

  const closeFolderPicker = () => {
    setFolderPicker((prev) => ({ ...prev, open: false }));
  };

  const selectFolder = (path) => {
    if (!path) return;
    if (folderPicker.target === "input") {
      setCentroidInputDir(path);
    } else {
      setCentroidOutDir(path);
    }
    closeFolderPicker();
  };

  const startCentroidRanking = async () => {
    if (!centroidInputDir.trim()) {
      toast.error("Karışık fotoğraf klasörünü seçin");
      return;
    }

    setCentroidStarting(true);
    setCentroidJob(null);
    try {
      const res = await axios.post(`${API}/dataset/centroid-rank`, {
        input_dir: centroidInputDir.trim(),
        out_dir: centroidOutDir.trim() || null,
        recursive: centroidRecursive,
        copy_mode: centroidCopyMode,
        bands: 3
      });
      setCentroidJob(res.data);
      toast.success("Centroid temizleme işi başlatıldı");
    } catch (error) {
      console.error("Centroid ranking failed to start", error);
      toast.error(error.response?.data?.detail || "Centroid temizleme başlatılamadı");
    } finally {
      setCentroidStarting(false);
    }
  };

  const cancelCentroidRanking = async () => {
    if (!centroidJob?.job_id || !centroidCanCancel) {
      return;
    }

    try {
      const res = await axios.post(`${API}/dataset/centroid-rank/${centroidJob.job_id}/cancel`);
      setCentroidJob(res.data);
      toast.info("Centroid temizleme durduruluyor");
    } catch (error) {
      console.error("Centroid ranking cancel failed", error);
      toast.error(error.response?.data?.detail || "Centroid temizleme durdurulamadı");
    }
  };

  const handleFileChange = (event) => {
    const selected = Array.from(event.target.files || []);
    if (!selected.length) return;

    const prepared = selected.map((file) => {
      const detected = detectPluFromName(file.name);
      const isValidDetected = pluList.some((plu) => plu.plu_code === detected);
      return {
        id: `${file.name}-${file.size}-${file.lastModified}-${Math.random().toString(36).slice(2, 6)}`,
        file,
        expectedPlu: isValidDetected ? detected : defaultPlu || "",
        detectedPlu: detected,
        status: "pending",
        result: null,
        error: null
      };
    });

    setFiles(prepared);
  };

  const updateExpectedPlu = (id, value) => {
    setFiles((prev) =>
      prev.map((item) =>
        item.id === id ? { ...item, expectedPlu: value } : item
      )
    );
  };

  const applyDefaultToAll = () => {
    if (!defaultPlu) {
      toast.error("Varsayılan PLU seçin");
      return;
    }
    setFiles((prev) =>
      prev.map((item) => ({
        ...item,
        expectedPlu: item.expectedPlu || defaultPlu
      }))
    );
    toast.success("Varsayılan PLU boş kalanlara uygulandı");
  };

  const resetSelection = () => {
    setFiles([]);
    setSummary(null);
    setFolderingInfo(null);
  };

  const startProcessing = async () => {
    if (!files.length) {
      toast.error("Önce fotoğraf yükleyin");
      return;
    }

    const missing = files.filter((f) => !(f.expectedPlu || defaultPlu));
    if (missing.length) {
      toast.error("Her fotoğraf için beklenen PLU'yu seçin");
      return;
    }

    setProcessing(true);
    setSummary(null);
    setFolderingInfo(null);
    setFiles((prev) => prev.map((item) => ({ ...item, status: "processing" })));

    try {
      const metadata = files.map((item) => ({
        filename: item.file.name,
        plu_code: item.expectedPlu || defaultPlu
      }));

      const formData = new FormData();
      files.forEach((item) => formData.append("files", item.file, item.file.name));
      formData.append("metadata", JSON.stringify(metadata));
      formData.append("foldering_enabled", folderingEnabled ? "true" : "false");

      const res = await axios.post(`${API}/batch/validate`, formData, {
        headers: { "Content-Type": "multipart/form-data" }
      });

      const resultList = res.data?.results || [];
      const summaryData = res.data?.summary || null;
      const nextFoldering = res.data?.foldering || null;

      setFiles((prev) =>
        prev.map((item) => {
          const result = resultList.find((r) => r.filename === item.file.name);
          if (!result) {
            return { ...item, status: "error", error: "Sonuç alınamadı" };
          }
          const status = result.status || (result.error ? "error" : result.is_match ? "match" : "mismatch");
          return {
            ...item,
            status,
            result,
            error: result.error || null
          };
        })
      );

      setSummary(summaryData);
      setFolderingInfo(nextFoldering);
      toast.success("Toplu kontrol tamamlandı");
      if (summaryData?.error_count) {
        toast.info(`${summaryData.error_count} fotoğraf işlenemedi`);
      }
    } catch (error) {
      console.error("Batch validation failed", error);
      toast.error("Toplu kontrol başarısız");
      setFolderingInfo(null);
      setFiles((prev) =>
        prev.map((item) => ({
          ...item,
          status: item.status === "processing" ? "error" : item.status
        }))
      );
    } finally {
      setProcessing(false);
    }
  };

  return (
    <div className="page-container batch-processing" data-testid="batch-processing-page">
      <div className="batch-header">
        <div>
          <h1 data-testid="batch-title">Toplu İşleme</h1>
          <p className="subtitle" data-testid="batch-subtitle">
            Toplu gelen fotoğrafları tek seferde kontrol edip uyumlu/uyumsuz sonucunu alın.
          </p>
        </div>
        <div className="batch-actions">
          <Button
            variant="outline"
            onClick={resetSelection}
            disabled={!files.length || processing}
            data-testid="batch-reset-btn"
          >
            Temizle
          </Button>
          <Button
            onClick={startProcessing}
            disabled={!files.length || processing}
            data-testid="batch-start-btn"
          >
            {processing ? "İşleniyor..." : "Analizi Başlat"}
          </Button>
        </div>
      </div>

      <Card className="batch-info-card">
        <CardHeader>
          <CardTitle>Nasıl çalışır?</CardTitle>
          <CardDescription>
            Fotoğrafları ekleyin, beklenen PLU kodunu tanımlayın, tek tuşla tümünü kontrol edin.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ol className="batch-steps">
            <li>Fotoğrafları seçin veya sürükleyip bırakın. Dosya adındaki rakamları otomatik PLU önerisi olarak okuyoruz (örn: 101_kasap.jpg).</li>
            <li>Gerekirse beklenen PLU'yu her fotoğraf için düzeltin ya da varsayılan PLU'yu boşlara uygulayın.</li>
            <li>"Analizi Başlat" ile tüm fotoğraflar sırasıyla işlenir ve uyumlu/uyumsuz sonucu oluşturulur.</li>
          </ol>
        </CardContent>
      </Card>

      <Card className="centroid-card">
        <CardHeader>
          <CardTitle>Centroid Temizleme</CardTitle>
          <CardDescription>
            Backend'in erişebildiği bir klasördeki karışık fotoğrafları ResNet centroid'ine yakınlığa göre sıralar.
          </CardDescription>
        </CardHeader>
        <CardContent className="centroid-grid">
          <div className="centroid-field">
            <Label htmlFor="centroid-input-dir">Karışık fotoğraf klasörü</Label>
            <div className="folder-select-row" id="centroid-input-dir">
              <div className={`folder-selected-path ${centroidInputDir ? "" : "empty"}`}>
                {centroidInputDir || "Klasör seçilmedi"}
              </div>
              <Button
                type="button"
                variant="outline"
                onClick={() => openFolderPicker("input")}
                disabled={centroidBusy}
              >
                Seç
              </Button>
            </div>
          </div>
          <div className="centroid-field">
            <Label htmlFor="centroid-out-dir">Çıktı klasörü</Label>
            <div className="folder-select-row" id="centroid-out-dir">
              <div className={`folder-selected-path ${centroidOutDir ? "" : "empty"}`}>
                {centroidOutDir || "Boşsa backend\\centroid_rank_jobs altında oluşturulur"}
              </div>
              <Button
                type="button"
                variant="outline"
                onClick={() => openFolderPicker("output")}
                disabled={centroidBusy}
              >
                Seç
              </Button>
              {centroidOutDir && (
                <Button
                  type="button"
                  variant="ghost"
                  onClick={() => setCentroidOutDir("")}
                  disabled={centroidBusy}
                >
                  Temizle
                </Button>
              )}
            </div>
          </div>
          <div className="centroid-options">
            <label className="foldering-toggle" htmlFor="centroid-recursive">
              <input
                id="centroid-recursive"
                type="checkbox"
                checked={centroidRecursive}
                onChange={(e) => setCentroidRecursive(e.target.checked)}
                disabled={centroidBusy}
              />
              <span>Alt klasörleri de tara</span>
            </label>
            <div className="centroid-field compact">
              <Label htmlFor="centroid-copy-mode">Kopyalama</Label>
              <select
                id="centroid-copy-mode"
                value={centroidCopyMode}
                onChange={(e) => setCentroidCopyMode(e.target.value)}
                disabled={centroidBusy}
              >
                <option value="bands">Band klasörleri</option>
                <option value="ranked">Tek sıralı klasör</option>
                <option value="none">Sadece CSV</option>
              </select>
            </div>
            <div className="centroid-band-info">
              <strong>Band:</strong> 3 adaptive klasör
              <span>01_nearest, 02_review, 03_farthest</span>
            </div>
          </div>
          <div className="centroid-actions">
            <Button
              onClick={startCentroidRanking}
              disabled={centroidBusy}
            >
              {centroidBusy ? "Çalışıyor..." : "Centroid Sıralamayı Başlat"}
            </Button>
            {centroidCanCancel && (
              <Button
                type="button"
                variant="outline"
                onClick={cancelCentroidRanking}
              >
                Durdur
              </Button>
            )}
          </div>
        </CardContent>
        {centroidJob && (
          <CardContent className="centroid-status">
            <p>
              <strong>Durum:</strong> {centroidStatusCopy[centroidJob.status] || centroidJob.status}
              {centroidJob.total_count > 0 && (
                <> | <strong>İlerleme:</strong> {centroidJob.processed_count || 0} / {centroidJob.total_count}</>
              )}
            </p>
            {centroidJob.output_dir && (
              <p><strong>Çıktı:</strong> {centroidJob.output_dir}</p>
            )}
            {centroidJob.ranking_csv && (
              <p><strong>CSV:</strong> {centroidJob.ranking_csv}</p>
            )}
            {centroidJob.status === "complete" && centroidJob.result && (
              <p>
                <strong>Özet:</strong> {centroidJob.result.processed_count} işlendi,
                {" "}{centroidJob.result.skipped_count} atlandı,
                {" "}similarity {centroidJob.result.similarity_min} - {centroidJob.result.similarity_max}
              </p>
            )}
            {centroidJob.status === "complete" && centroidJob.result?.band_thresholds && (
              <p>
                <strong>Adaptive sınırlar:</strong>
                {" "}review &lt; {centroidJob.result.band_thresholds.review_if_below},
                {" "}farthest &lt; {centroidJob.result.band_thresholds.farthest_if_below}
              </p>
            )}
            {centroidJob.status === "complete" && centroidJob.result?.band_counts && (
              <p>
                <strong>Band adetleri:</strong>
                {" "}
                {Object.entries(centroidJob.result.band_counts)
                  .map(([name, count]) => `${name}: ${count}`)
                  .join(" | ")}
              </p>
            )}
            {centroidJob.status === "error" && (
              <p className="centroid-error">{centroidJob.error || "Bilinmeyen hata"}</p>
            )}
          </CardContent>
        )}
      </Card>

      {folderPicker.open && (
        <div className="folder-picker-overlay" onClick={closeFolderPicker}>
          <div className="folder-picker-modal" onClick={(event) => event.stopPropagation()}>
            <div className="folder-picker-header">
              <div>
                <h3>{folderPicker.title}</h3>
                <p>Backend'in erişebildiği klasörler listelenir. Klasöre girmek için Aç, seçmek için Seç kullanın.</p>
              </div>
              <button type="button" className="folder-picker-close" onClick={closeFolderPicker}>
                ×
              </button>
            </div>

            <div className="folder-picker-toolbar">
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() => fetchFolderList("")}
                disabled={folderPicker.loading}
              >
                Kökler
              </Button>
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() => fetchFolderList(folderPicker.parentPath)}
                disabled={!folderPicker.parentPath || folderPicker.loading}
              >
                Üst Klasör
              </Button>
              {folderPicker.currentPath && (
                <Button
                  type="button"
                  size="sm"
                  onClick={() => selectFolder(folderPicker.currentPath)}
                  disabled={folderPicker.loading}
                >
                  Bu Klasörü Seç
                </Button>
              )}
            </div>

            <div className="folder-picker-current">
              <strong>Konum:</strong> {folderPicker.currentPath || "Kök klasörler"}
            </div>

            {folderPicker.error && (
              <div className="folder-picker-error">{folderPicker.error}</div>
            )}

            <div className="folder-picker-list">
              {folderPicker.loading ? (
                <div className="folder-picker-empty">Klasörler yükleniyor...</div>
              ) : folderPicker.entries.length === 0 ? (
                <div className="folder-picker-empty">Alt klasör bulunamadı.</div>
              ) : (
                folderPicker.entries.map((entry) => (
                  <div className="folder-picker-entry" key={entry.path}>
                    <button
                      type="button"
                      className="folder-picker-entry-main"
                      onClick={() => fetchFolderList(entry.path)}
                    >
                      <span className="folder-picker-entry-name">{entry.name}</span>
                      <span className="folder-picker-entry-path">{entry.path}</span>
                    </button>
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      onClick={() => selectFolder(entry.path)}
                    >
                      Seç
                    </Button>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      )}

      <Card className="upload-card">
        <CardHeader>
          <CardTitle>Fotoğraf Yükleme</CardTitle>
          <CardDescription>Toplu test için aynı anda birden fazla görsel ekleyin.</CardDescription>
        </CardHeader>
        <CardContent className="upload-grid">
          <div className="upload-zone">
            <Label htmlFor="batch-files">Fotoğraflar</Label>
            <label className="drop-area" htmlFor="batch-files" data-testid="drop-area">
              <input
                id="batch-files"
                type="file"
                accept="image/*"
                multiple
                onChange={handleFileChange}
                disabled={processing}
                data-testid="batch-file-input"
              />
              <div>
                <p className="drop-title">Sürükle &amp; bırak veya tıkla</p>
                <p className="drop-subtitle">JPG, PNG veya WEBP formatında birden fazla dosya ekleyebilirsiniz.</p>
              </div>
            </label>
          </div>

          <div className="default-plu">
            <Label htmlFor="default-plu">Varsayılan PLU (isteğe bağlı)</Label>
            <div className="default-row">
              <select
                id="default-plu"
                value={defaultPlu}
                onChange={(e) => setDefaultPlu(e.target.value)}
                disabled={processing}
                data-testid="default-plu-select"
              >
                <option value="">Seçilmedi</option>
                {pluList.map((plu) => (
                  <option key={plu.id} value={plu.plu_code}>
                    {plu.plu_code} - {plu.name}
                  </option>
                ))}
              </select>
              <Button
                variant="outline"
                onClick={applyDefaultToAll}
                disabled={!files.length || processing}
                data-testid="apply-default-btn"
              >
                Boş kalanlara uygula
              </Button>
            </div>
            <p className="helper-text">
              Dosya adında PLU varsa otomatik eşleştiriyoruz, yoksa buradan seçebilirsiniz.
            </p>
            <label className="foldering-toggle" htmlFor="foldering-enabled">
              <input
                id="foldering-enabled"
                type="checkbox"
                checked={folderingEnabled}
                onChange={(e) => setFolderingEnabled(e.target.checked)}
                disabled={processing}
              />
              <span>Klasörleme Aktif</span>
            </label>
            <p className="helper-text">
              Aktif olursa sonuç dosyaları
              {" "}
              <code>backend\\batch_foldering\\&lt;batch_id&gt;</code>
              {" "}
              altına benzerlik kategorilerine göre kopyalanır.
            </p>
          </div>
        </CardContent>
      </Card>

      <Card className="file-list-card">
        <CardHeader>
          <CardTitle>Seçilen Fotoğraflar</CardTitle>
          <CardDescription>Beklenen PLU ve işleme durumu.</CardDescription>
        </CardHeader>
        <CardContent>
          {files.length === 0 ? (
            <div className="empty-placeholder" data-testid="batch-empty-state">
              <p>Henüz fotoğraf eklenmedi.</p>
              <p className="helper-text">Yukarıdan dosya ekleyin.</p>
            </div>
          ) : (
            <div className="file-table" data-testid="batch-file-table">
              <div className="file-row header">
                <div>Dosya</div>
                <div>Beklenen PLU</div>
                <div>Durum</div>
                <div>Güven</div>
              </div>
              {files.map((item) => (
                <div className="file-row" key={item.id}>
                  <div className="file-name">
                    <div className="file-title">{item.file.name}</div>
                    <div className="file-meta">{formatSize(item.file.size)}</div>
                    {item.detectedPlu && (
                      <span className="detected-tag">Öneri: {item.detectedPlu}</span>
                    )}
                  </div>
                  <div className="file-plu">
                    <select
                      value={item.expectedPlu || ""}
                      onChange={(e) => updateExpectedPlu(item.id, e.target.value)}
                      disabled={processing}
                      data-testid={`expected-plu-${item.id}`}
                    >
                      <option value="">PLU seçin</option>
                      {pluList.map((plu) => (
                        <option key={plu.id} value={plu.plu_code}>
                          {plu.plu_code} - {plu.name}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className={`status-chip ${item.status}`} data-testid={`status-${item.id}`}>
                    {statusCopy[item.status] || "Bekliyor"}
                  </div>
                  <div className="confidence-cell" data-testid={`confidence-${item.id}`}>
                    {item.result && !item.error ? `${item.result.confidence}%` : "-"}
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {summary && (
        <Card className="summary-card" data-testid="batch-summary">
          <CardHeader>
            <CardTitle>Sonuç Özeti</CardTitle>
            <CardDescription>Toplam {summary.processed} fotoğraf işlendi.</CardDescription>
          </CardHeader>
          <CardContent className="summary-grid">
            <div>
              <p className="summary-value">{summary.total}</p>
              <p className="summary-label">Toplam</p>
            </div>
            <div>
              <p className="summary-value ok">{summary.match_count}</p>
              <p className="summary-label">Uyumlu</p>
            </div>
            <div>
              <p className="summary-value warn">{summary.mismatch_count}</p>
              <p className="summary-label">Uyumsuz</p>
            </div>
            <div>
              <p className="summary-value error">{summary.error_count}</p>
              <p className="summary-label">Hatalı</p>
            </div>
          </CardContent>
          {folderingInfo?.enabled && (
            <CardContent className="foldering-summary">
              <p>
                <strong>Klasör yolu:</strong> {folderingInfo.base_dir || "-"}
              </p>
              <p>
                <strong>Kopyalanan:</strong> {folderingInfo.created_files || 0}
                {" | "}
                <strong>Hata:</strong> {folderingInfo.failed_files || 0}
              </p>
              <p>
                <strong>Eşikler:</strong>
                {" "}
                düşük &lt; {folderingInfo.low_threshold_pct}%
                {" | "}
                yüksek &gt;= {folderingInfo.high_threshold_pct}%
              </p>
              {folderingInfo.categories && Object.keys(folderingInfo.categories).length > 0 && (
                <p>
                  <strong>Kategoriler:</strong>
                  {" "}
                  {Object.entries(folderingInfo.categories)
                    .map(([k, v]) => `${k}: ${v}`)
                    .join(" | ")}
                </p>
              )}
            </CardContent>
          )}
        </Card>
      )}

      {files.some((f) => f.result || f.error) && (
        <Card className="results-card">
          <CardHeader>
            <CardTitle>Detaylı Sonuçlar</CardTitle>
            <CardDescription>Her fotoğraf için model çıktısı</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="result-list">
              {files.map((item) => (
                <div key={item.id} className={`result-item ${item.status}`}>
                  <div>
                    <div className="result-title">{item.file.name}</div>
                    <p className="result-subtitle">
                      Beklenen PLU: {item.expectedPlu || defaultPlu || "-"}
                    </p>
                    <p className="analysis-text">
                      {item.error
                        ? item.error
                        : item.result?.analysis || "Analiz sonucu bekleniyor."}
                    </p>
                    {!item.error && Array.isArray(item.result?.top_matches) && item.result.top_matches.length > 0 && (
                      <p className="analysis-text">
                        Top {topMatchesLimitFor(item.result)} benzer: {summarizeTopMatches(item.result.top_matches, topMatchesLimitFor(item.result))}
                      </p>
                    )}
                    {!item.error && item.result?.folder_category && (
                      <p className="analysis-text">
                        Klasör kategorisi: {item.result.folder_category}
                      </p>
                    )}
                  </div>
                  <div className="result-meta">
                    <span className={`result-badge ${item.status}`}>
                      {item.error
                        ? "Hata"
                        : item.status === "match"
                        ? "Uyumlu"
                        : item.status === "mismatch"
                        ? "Uyumsuz"
                        : statusCopy[item.status] || "Bekliyor"}
                    </span>
                    {item.result?.processing_ms !== undefined && (
                      <span className="result-time">{item.result.processing_ms} ms</span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default BatchProcessing;
