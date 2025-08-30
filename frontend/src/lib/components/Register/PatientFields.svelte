<script lang="ts">
    export let formData: any;
</script>

<select
    bind:value={formData.sex}
    name="sex"
    class="input"
    required
>
    <option value="" disabled>Select Gender</option>
    <option value="male">Male</option>
    <option value="female">Female</option>
    <option value="other">Other</option>
    <option value="prefer_not_to_say">Prefer not to say</option>
</select>

<input
    type="date"
    bind:value={formData.dob}
    name="date_of_birth"
    class="input"
    required
/>

<input
    bind:value={formData.contact_info}
    placeholder="Phone Number"
    name="phone_number"
    type="tel"
    class="input"
    required
/>

<input
    type="date"
    bind:value={formData.medical_record_number}
    name="date_of_birth"
    class="input"
    required
/>