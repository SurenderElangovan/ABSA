	function openPopup() {
      var popupContainer = document.getElementById('popup-container');
      
      var popup = document.createElement('div');
      popup.className = 'popup';
      popup.innerHTML = '<div class="popup-content">No new notification</div><span class="popup-close" onclick="this.parentNode.remove();">&times;</span>';
      popupContainer.appendChild(popup);
      
      setTimeout(function() {
        popup.remove();
      }, 8000); // Remove the popup after 8 seconds
    }
	$("#Analytics").click(function () {
		$.ajax({
			url: "/Analytics",
			type: "GET",
			success: function (data) {
				$("#content").html(data);
			}
		});
	});
	$("#Analytics").click(function () {
		$.ajax({
			url: "/home",
			type: "GET",
			success: function (data) {
				$("#content").html(data);
			}
		});
	});