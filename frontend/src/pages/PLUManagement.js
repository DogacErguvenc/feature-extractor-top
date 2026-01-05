import React, { useState, useEffect } from "react";
import axios from "axios";
import { API } from "@/App";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "sonner";
import "./PLUManagement.css";

const PLUManagement = () => {
  const [plusList, setPlusList] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showAddForm, setShowAddForm] = useState(false);
  const [formData, setFormData] = useState({
    plu_code: "",
    name: "",
    description: ""
  });

  useEffect(() => {
    fetchPLUList();
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

  const handleInputChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!formData.plu_code || !formData.name || !formData.description) {
      toast.error("Tüm alanları doldurun");
      return;
    }

    try {
      await axios.post(`${API}/plu/create`, formData);
      toast.success("PLU başarıyla eklendi!");
      setFormData({ plu_code: "", name: "", description: "" });
      setShowAddForm(false);
      fetchPLUList();
    } catch (error) {
      console.error("Error creating PLU:", error);
      toast.error("PLU eklenemedi: " + (error.response?.data?.detail || "Hata"));
    }
  };

  const handleDelete = async (plu_code) => {
    if (!window.confirm(`PLU ${plu_code} silinecek. Emin misiniz?`)) {
      return;
    }

    try {
      await axios.delete(`${API}/plu/delete/${plu_code}`);
      toast.success("PLU silindi");
      fetchPLUList();
    } catch (error) {
      console.error("Error deleting PLU:", error);
      toast.error("PLU silinemedi");
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
    <div className="page-container plu-management" data-testid="plu-management-page">
      <div className="plu-header">
        <div>
          <h1 data-testid="page-title">PLU Yönetimi</h1>
          <p className="subtitle" data-testid="page-subtitle">Ürün kodlarını yönetin</p>
        </div>
        <Button 
          onClick={() => setShowAddForm(!showAddForm)}
          data-testid="toggle-add-form-btn"
        >
          {showAddForm ? "❌ İptal" : "➕ Yeni PLU Ekle"}
        </Button>
      </div>

      {showAddForm && (
        <Card className="add-form-card" data-testid="add-plu-form">
          <CardHeader>
            <CardTitle>Yeni PLU Ekle</CardTitle>
            <CardDescription>Ürün bilgilerini girin</CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="plu-form">
              <div className="form-group">
                <Label htmlFor="plu_code">PLU Kodu</Label>
                <Input
                  id="plu_code"
                  name="plu_code"
                  placeholder="Örn: 101"
                  value={formData.plu_code}
                  onChange={handleInputChange}
                  data-testid="plu-code-input"
                />
              </div>
              
              <div className="form-group">
                <Label htmlFor="name">Ürün Adı</Label>
                <Input
                  id="name"
                  name="name"
                  placeholder="Örn: Dana Kıyma"
                  value={formData.name}
                  onChange={handleInputChange}
                  data-testid="plu-name-input"
                />
              </div>
              
              <div className="form-group">
                <Label htmlFor="description">Açıklama</Label>
                <Textarea
                  id="description"
                  name="description"
                  placeholder="Ürün detaylarını girin (renk, şekil, özellikler)"
                  value={formData.description}
                  onChange={handleInputChange}
                  rows={4}
                  data-testid="plu-description-input"
                />
              </div>
              
              <Button type="submit" className="submit-btn" data-testid="submit-plu-btn">
                ✓ PLU Ekle
              </Button>
            </form>
          </CardContent>
        </Card>
      )}

      <div className="plu-list" data-testid="plu-list">
        {plusList.length === 0 ? (
          <Card className="empty-state" data-testid="empty-plu-state">
            <CardContent className="empty-content">
              <div className="empty-icon">🏷️</div>
              <h3>Henüz PLU eklenmemiş</h3>
              <p>Başlamak için yukarıdaki butona tıklayın</p>
            </CardContent>
          </Card>
        ) : (
          <div className="plu-cards-grid">
            {plusList.map((plu) => (
              <Card key={plu.id} className="plu-item-card" data-testid={`plu-item-${plu.plu_code}`}>
                <CardHeader>
                  <CardTitle data-testid={`plu-item-title-${plu.plu_code}`}>
                    <span className="plu-code">PLU {plu.plu_code}</span>
                  </CardTitle>
                  <CardDescription data-testid={`plu-item-name-${plu.plu_code}`}>{plu.name}</CardDescription>
                </CardHeader>
                <CardContent>
                  <p className="plu-desc" data-testid={`plu-item-desc-${plu.plu_code}`}>{plu.description}</p>
                  <div className="plu-meta" data-testid={`plu-item-date-${plu.plu_code}`}>
                    Eklenme: {new Date(plu.created_at).toLocaleDateString('tr-TR')}
                  </div>
                  <Button 
                    variant="destructive" 
                    className="delete-btn"
                    onClick={() => handleDelete(plu.plu_code)}
                    data-testid={`delete-plu-btn-${plu.plu_code}`}
                  >
                    🗑️ Sil
                  </Button>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default PLUManagement;