<script>
  let message = "";
  
  // check dashboard api based on role
  async function fetchDashboard() {
    try {
      const role = localStorage.getItem("role");
      const response = await fetch(`http://127.0.0.1:8000/api/${role}/dashboard/`, {
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${localStorage.getItem("access_token")}`
        }
      });
      const data = await response.json();
      message = data.message;
      console.log(data);
      
    } catch (error) {
      message = "Error fetching dashboard data.";
    }
  }

  fetchDashboard();
</script>

<h1>{message}</h1>
