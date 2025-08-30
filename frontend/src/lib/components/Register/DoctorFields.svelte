<script lang="ts">
    export let formData: any;
</script>


<input
    bind:value={formData.license_number}
    placeholder="Medical License Number"
    name="license_number"
    class="input"
    required
/>

<input
    bind:value={formData.specialization}
    placeholder="Specialization (e.g., Cardiology, Neurology)"
    name="specialization"
    class="input"
    required
/>