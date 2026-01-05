import { useState, useEffect } from "react";
import "@/App.css";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import axios from "axios";
import ScaleOperator from "@/pages/ScaleOperator";
import Dashboard from "@/pages/Dashboard";
import PLUManagement from "@/pages/PLUManagement";
import BatchProcessing from "@/pages/BatchProcessing";
import Navigation from "@/components/Navigation";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;
const API_KEY = process.env.REACT_APP_API_KEY;

// Set default auth header for all axios requests
if (API_KEY) {
  axios.defaults.headers.common["x-api-key"] = API_KEY;
}

export { API };

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Navigation />
        <Routes>
          <Route path="/" element={<ScaleOperator />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/plu-management" element={<PLUManagement />} />
          <Route path="/batch-processing" element={<BatchProcessing />} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;
