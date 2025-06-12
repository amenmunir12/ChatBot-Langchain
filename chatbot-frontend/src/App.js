import React from 'react';
import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom';
import Home from './pages/home';
import Chat from './pages/chat';

function App() {
  return (
    <Router>
      <div style={styles.app}>
        <nav style={styles.navbar}>
          <NavLink
            to="/"
            style={({ isActive }) => ({
              ...styles.link,
              ...(isActive ? styles.activeLink : {})
            })}
          >
            Home
          </NavLink>
          <NavLink
            to="/chat"
            style={({ isActive }) => ({
              ...styles.link,
              ...(isActive ? styles.activeLink : {})
            })}
          >
            Chat
          </NavLink>
        </nav>

        <main style={styles.main}>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/chat" element={<Chat />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

const styles = {
  app: {
    fontFamily: 'Inter, sans-serif',
    minHeight: '100vh',
    backgroundColor: '#1e1e2f',
    color: '#fff'
  },
  navbar: {
    display: 'flex',
    gap: '20px',
    padding: '16px 32px',
    backgroundColor: '#2a2f4c',
    borderBottom: '1px solid #444',
    justifyContent: 'center',
    position: 'sticky',
    top: 0,
    zIndex: 1000
  },
  link: {
    textDecoration: 'none',
    color: '#ccc',
    fontWeight: '500',
    padding: '8px 16px',
    borderRadius: '8px',
    transition: 'background 0.2s ease, color 0.2s ease'
  },
  activeLink: {
    backgroundColor: '#4e8cff',
    color: '#fff'
  },
  main: {
    padding: '24px',
    maxWidth: '1200px',
    margin: '0 auto'
  }
};

export default App;
