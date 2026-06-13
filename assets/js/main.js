  function showScene(sceneId, btn) {
    // Hide all panels
    document.querySelectorAll('.comparison-panel').forEach(p => p.classList.remove('active'));
    // Deactivate all tabs
    document.querySelectorAll('.scene-tab').forEach(t => t.classList.remove('active'));
    // Show selected
    document.getElementById('panel-' + sceneId).classList.add('active');
    btn.classList.add('active');
  }
