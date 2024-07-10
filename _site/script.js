document.addEventListener("DOMContentLoaded", function() {
    var toggleButton = document.getElementById("toggle-button");
    var olderNews = document.getElementById("older-news");
  
    toggleButton.addEventListener("click", function() {
      if (olderNews.style.display === "none") {
        olderNews.style.display = "block";
        toggleButton.textContent = "Hide older news";
      } else {
        olderNews.style.display = "none";
        toggleButton.textContent = "Show older news";
      }
    });
  });