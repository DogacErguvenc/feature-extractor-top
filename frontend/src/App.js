import "@/App.css";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import ScaleOperator from "@/pages/ScaleOperator";
import Dashboard from "@/pages/Dashboard";
import PLUManagement from "@/pages/PLUManagement";
import BatchProcessing from "@/pages/BatchProcessing";
import ResnetAnalysis from "@/pages/ResnetAnalysis";
import Navigation from "@/components/Navigation";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;
export { API };

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Navigation />
        <Routes>
          <Route path="/" element={<ScaleOperator />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/analysis" element={<ResnetAnalysis />} />
          <Route path="/plu-management" element={<PLUManagement />} />
          <Route path="/batch-processing" element={<BatchProcessing />} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;

