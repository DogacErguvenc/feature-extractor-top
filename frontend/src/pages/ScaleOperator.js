import React, { useState, useEffect } from "react";
import axios from "axios";
import { API } from "@/App";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { toast } from "sonner";
import "./ScaleOperator.css";

const ScaleOperator = () => {
  const [plusList, setPlusList] = useState([]);
  const [loading, setLoading] = useState(true);
  const [systemMode, setSystemMode] = useState("training");
  const [processing, setProcessing] = useState(false);
  const [selectedPLU, setSelectedPLU] = useState(null);

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
      toast.error("PLU listesi yüklenemedi");
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
        plu_code: plu.plu_code
      });
      
      toast.success(`✅ ${plu.name} seçildi ve fotoğraf çekildi!`);
      
      if (systemMode === "training") {
        toast.info(`📸 Toplanan fotoğraf sayısı: ${response.data.images_collected || 1}`);
      } else {
        toast.info("🤖 AI analizi başlatıldı...");
      }
      
      setTimeout(() => {
        setSelectedPLU(null);
      }, 2000);
    } catch (error) {
      console.error("Error selecting PLU:", error);
      toast.error("Hata: " + (error.response?.data?.detail || "Fotoğraf çekilemedi"));
      setSelectedPLU(null);
    } finally {
      setProcessing(false);
    }
  };

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
          <h1 data-testid="page-title">Terazi Operatör Ekranı</h1>
          <p className="subtitle" data-testid="page-subtitle">Lütfen ürün için PLU seçimi yapın</p>
        </div>
        <div className={`mode-badge ${systemMode}`} data-testid="system-mode-badge">
          <span className="mode-indicator"></span>
          {systemMode === "training" ? "Eğitim Modu" : "Üretim Modu"}
        </div>
      </div>

      {plusList.length === 0 ? (
        <Card className="empty-state" data-testid="empty-plu-state">
          <CardContent className="empty-content">
            <div className="empty-icon">🏷️</div>
            <h3>Henüz PLU eklenmemiş</h3>
            <p>Başlamak için PLU Yönetimi sayfasından ürün ekleyin</p>
          </CardContent>
        </Card>
      ) : (
        <div className="plu-grid" data-testid="plu-grid">
          {plusList.map((plu) => (
            <Card 
              key={plu.id} 
              className={`plu-card ${selectedPLU?.id === plu.id ? 'selected' : ''} ${processing && selectedPLU?.id === plu.id ? 'processing' : ''}`}
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
                    <span>⏳ İşleniyor...</span>
                  ) : (
                    <span>✓ Seç</span>
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
            <div className="info-icon">ℹ️</div>
            <div>
              <h3>Eğitim Modu Aktif</h3>
              <p>Sistem şu anda veri toplama modunda. Her PLU seçiminde fotoğraf çekiliyor ve kaydediliyor. 30 gün sonra yeterli veri toplandığında üretim moduna geçebilirsiniz.</p>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default ScaleOperator;