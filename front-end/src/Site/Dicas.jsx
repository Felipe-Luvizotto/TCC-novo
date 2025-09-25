import './Dicas.css';
import { Link } from 'react-router-dom';
import { ShieldAlert, CloudRain, MapPin, PhoneCall } from 'lucide-react';

const Dicas = () => {
  return (
    <section className="dicas-section">
      <Link to="/" className="back-link">←</Link>
      <div className="dicas-container">
        <h1>🚨 O que fazer em caso de enchente</h1>
        <p className="intro-text">
          Em situações de enchente, manter a calma e seguir orientações corretas pode salvar vidas. Confira abaixo algumas ações importantes para garantir sua segurança e de sua família:
        </p>

        <div className="cards-grid">
          <div className="card">
            <ShieldAlert size={32} />
            <h2>Desligue equipamentos elétricos</h2>
            <p>Evite choques elétricos desligando a energia da casa ao perceber risco de inundação.</p>
          </div>

          <div className="card">
            <MapPin size={32} />
            <h2>Evacue áreas de risco</h2>
            <p>Se você mora próximo a rios ou encostas, procure abrigo em locais altos e seguros.</p>
          </div>

          <div className="card">
            <CloudRain size={32} />
            <h2>Evite contato com a água</h2>
            <p>Águas de enchente podem estar contaminadas e esconder perigos como buracos e objetos cortantes.</p>
          </div>

          <div className="card">
            <PhoneCall size={32} />
            <h2>Comunique as autoridades</h2>
            <p>Ligue para a Defesa Civil (199) ou Corpo de Bombeiros (193) em caso de emergência.</p>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Dicas;
