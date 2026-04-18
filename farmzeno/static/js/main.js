// FarmZeno — main.js (shared utilities)

// ── Saved city ────────────────────────────────────────────
function getSavedCity() { try { return localStorage.getItem('fz_city') || ''; } catch(e){ return ''; } }
function saveCity(city) { try { if(city) localStorage.setItem('fz_city', city); } catch(e){} }

// ── DOM helpers ───────────────────────────────────────────
function show(id) { const el = document.getElementById(id); if(el) el.classList.remove('hidden'); }
function hide(id) { const el = document.getElementById(id); if(el) el.classList.add('hidden'); }
function showErr(id, msg) {
  const el = document.getElementById(id);
  if(el) { el.textContent = '⚠️ ' + msg; el.classList.remove('hidden'); }
}

// ── Mobile nav toggle ─────────────────────────────────────
document.addEventListener('DOMContentLoaded', function() {
  const toggle = document.getElementById('navToggle');
  const links  = document.getElementById('navLinks');
  if(toggle && links) {
    toggle.addEventListener('click', function(e) {
      e.stopPropagation();
      links.classList.toggle('open');
    });
    document.addEventListener('click', function(e) {
      if(!links.contains(e.target) && e.target !== toggle) {
        links.classList.remove('open');
      }
    });
    // Close on nav link click
    links.querySelectorAll('.nav-link').forEach(function(link) {
      link.addEventListener('click', function() { links.classList.remove('open'); });
    });
  }
});
