import React from 'react';

function Home() {
  const handleStart = () => {
    
    window.location.href = '/chat'; 
  };

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>Welcome to ChatFlow</h1>
      <p style={styles.description}>
        Chat securely, ask questions, or upload documents to get personalized responses.
      </p>
      <button style={styles.button} onClick={handleStart}>
        Start Chatting
      </button>
    </div>
  );
}

const styles = {
  container: {
    maxWidth: '700px',
    margin: '120px auto',
    padding: '40px',
    textAlign: 'center',
    fontFamily: "'Segoe UI', sans-serif",
    backgroundColor: '#1c1f2b',
    borderRadius: '16px',
    boxShadow: '0 0 25px rgba(0,0,0,0.3)',
    color: '#f1f1f1',
  },
  title: {
    fontSize: '2.8rem',
    fontWeight: '600',
    marginBottom: '20px',
    color: '#e1e6f0',
  },
  description: {
    fontSize: '1.2rem',
    color: '#c4c9d4',
    marginBottom: '40px',
    lineHeight: '1.6',
  },
  button: {
    backgroundColor: '#4d79f6',
    color: '#fff',
    border: 'none',
    padding: '14px 28px',
    fontSize: '1rem',
    borderRadius: '8px',
    cursor: 'pointer',
    transition: 'background 0.3s ease, transform 0.2s ease',
  },
};

export default Home;
