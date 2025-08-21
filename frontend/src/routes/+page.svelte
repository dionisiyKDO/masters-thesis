<script>
  let message = "";
  fetch("http://127.0.0.1:8000/api/hello/")
    .then(r => r.json())
    .then(data => message = data.message);
</script>

<h1>{message}</h1>
