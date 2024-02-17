let dropdownVisible = false;

function toggleDropdown() {
  const dropdownContent = document.querySelector(".dropdown-content");
  dropdownVisible = !dropdownVisible;
  const caretIcon = document.getElementById("caretIcon");
  if (dropdownVisible) {
    dropdownContent.style.display = "block";
    caretIcon.classList.remove("fa-caret-down");
    caretIcon.classList.add("fa-caret-up");
  } else {
    dropdownContent.style.display = "none";
    caretIcon.classList.remove("fa-caret-up");
    caretIcon.classList.add("fa-caret-down");
  }
}

document.addEventListener("click", function (event) {
  const isClickInsideButton = document
    .querySelector(".dropbtn")
    .contains(event.target);
  const isClickInsideDropdownContent = document
    .querySelector(".dropdown-content")
    .contains(event.target);
  if (!isClickInsideButton && !isClickInsideDropdownContent) {
    const dropdownContent = document.querySelector(".dropdown-content");
    dropdownContent.style.display = "none";
    dropdownVisible = false;
    const caretIcon = document.getElementById("caretIcon");
    caretIcon.classList.remove("fa-caret-up");
    caretIcon.classList.add("fa-caret-down");
  }
});

document.addEventListener("DOMContentLoaded", function () {
  // Get all dropdown items
  var dropdownItems = document.querySelectorAll(".dropdown-item");

  // Loop through each dropdown item
  dropdownItems.forEach(function (item) {
    // Add click event listener to each dropdown item
    item.addEventListener("click", function (event) {
      // Prevent default link behavior
      event.preventDefault();

      // Get the ID of the clicked item
      var itemId = event.target.getAttribute("id");

      // Check if the clicked item is "Air"
      if (itemId === "air_pollution_YearList") {
        window.location.href = "airDataSetYearList.html";
      } else if (itemId === "noiseDataSetYearList") {
        window.location.href = "noiseDataSetYearList.html";
      } else if (itemId === "waterDataSetYearList") {
        window.location.href = "waterDataSetYearList.html";
      } else if (itemId === "air_pollution_analysis") {
        window.location.href = "airDataAnalysis.html";
      } else if (itemId === "water_pollution_analysis") {
        window.location.href = "waterDataAnalysis.html";
      } else if (itemId === "noise_pollution_analysis") {
        window.location.href = "noiseDataAnalysis.html";
      } else {
        // Handle other dropdown items if needed
        console.log("Clicked item: " + itemId);
      }
    });
  });
});
