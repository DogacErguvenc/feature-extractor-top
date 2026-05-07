import React from "react";
import { Link, useLocation } from "react-router-dom";
import "./Navigation.css";

const Navigation = () => {
  const location = useLocation();
  const isActive = (path) => location.pathname === path;

  return (
    <nav className="navigation" data-testid="main-navigation">
      <div className="nav-container">
        <div className="nav-brand" data-testid="nav-brand">
          <span className="brand-icon" role="img" aria-label="terazi-logo">🏷️</span>
          <span className="brand-text">Terazi AI</span>
        </div>
        
        <div className="nav-links">
          <Link 
            to="/" 
            className={`nav-link ${isActive("/") ? "active" : ""}`}
            data-testid="nav-link-operator"
          >
            <span className="nav-icon" role="img" aria-label="scale-screen">🏪</span>
            Terazi Ekranı
          </Link>
          
          <Link 
            to="/dashboard" 
            className={`nav-link ${isActive("/dashboard") ? "active" : ""}`}
            data-testid="nav-link-dashboard"
          >
            <span className="nav-icon" role="img" aria-label="dashboard">📊</span>
            Dashboard
          </Link>

          <Link
            to="/analysis"
            className={`nav-link ${isActive("/analysis") ? "active" : ""}`}
            data-testid="nav-link-analysis"
          >
            <span className="nav-icon" aria-hidden="true">AN</span>
            Analiz
          </Link>
          
          <Link 
            to="/batch-processing" 
            className={`nav-link ${isActive("/batch-processing") ? "active" : ""}`}
            data-testid="nav-link-batch"
          >
            <span className="nav-icon" role="img" aria-label="batch-processing">📦</span>
            Toplu İşleme
          </Link>
          
          <Link 
            to="/plu-management" 
            className={`nav-link ${isActive("/plu-management") ? "active" : ""}`}
            data-testid="nav-link-plu"
          >
            <span className="nav-icon" role="img" aria-label="plu">🏷️</span>
            PLU Yönetimi
          </Link>
        </div>
      </div>
    </nav>
  );
};

export default Navigation;
