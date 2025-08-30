<script lang="ts">
    export let formData: any;
</script>

<div class="flex gap-2">
    <input
        bind:value={formData.first_name}
        placeholder="First Name"
        name="first_name"
        class="input"
        required
    />
    <input
        bind:value={formData.last_name}
        placeholder="Last Name"
        name="last_name"
        class="input"
        required
    />
</div>

<input
    bind:value={formData.email}
    type="email"
    placeholder="example@gmail.com"
    name="email"
    class="input"
    required
/>