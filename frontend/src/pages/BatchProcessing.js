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

const summarizeTopMatches = (matches) => {
  if (!Array.isArray(matches) || matches.length === 0) return "-";
  return matches
    .slice(0, 3)
    .map((match, idx) => {
      const rank = match?.rank ?? idx + 1;
      return `${rank}. ${match?.plu_code || "?"} (${formatTopMatchScore(match)})`;
    })
    .join(" | ");
};

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

const BatchProcessing = () => {
  const [pluList, setPluList] = useState([]);
  const [files, setFiles] = useState([]);
  const [defaultPlu, setDefaultPlu] = useState("");
  const [processing, setProcessing] = useState(false);
  const [summary, setSummary] = useState(null);

  useEffect(() => {
    fetchPluList();
  }, []);

  const fetchPluList = async () => {
    try {
      const res = await axios.get(`${API}/plu/list`);
      setPluList(res.data || []);
    } catch (error) {
      console.error("Error fetching PLU list", error);
      toast.error("PLU listesi alınamadı");
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
    setFiles((prev) => prev.map((item) => ({ ...item, status: "processing" })));

    try {
      const metadata = files.map((item) => ({
        filename: item.file.name,
        plu_code: item.expectedPlu || defaultPlu
      }));

      const formData = new FormData();
      files.forEach((item) => formData.append("files", item.file, item.file.name));
      formData.append("metadata", JSON.stringify(metadata));

      const res = await axios.post(`${API}/batch/validate`, formData, {
        headers: { "Content-Type": "multipart/form-data" }
      });

      const resultList = res.data?.results || [];
      const summaryData = res.data?.summary || null;

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
      toast.success("Toplu kontrol tamamlandı");
      if (summaryData?.error_count) {
        toast.info(`${summaryData.error_count} fotoğraf işlenemedi`);
      }
    } catch (error) {
      console.error("Batch validation failed", error);
      toast.error("Toplu kontrol başarısız");
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
                        Top 3 benzer: {summarizeTopMatches(item.result.top_matches)}
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
