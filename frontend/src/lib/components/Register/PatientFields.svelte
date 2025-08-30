<script lang="ts">
	import type { FormData } from "$lib/types";

    interface Props { formData: FormData }
    let { formData = $bindable() }: Props = $props()
</script>

<select
    bind:value={formData.patient_profile!.sex}
    name="sex"
    class="input"
    required
>
    <option value="" disabled>Select Gender</option>
    <option value="M">Male</option>
    <option value="F">Female</option>
    <option value="O">Other</option>
</select>

<input
    type="date"
    bind:value={formData.patient_profile!.dob}
    name="date_of_birth"
    class="input"
    required
/>

<input
    bind:value={formData.patient_profile!.contact_info}
    placeholder="Phone Number"
    name="phone_number"
    type="tel"
    class="input"
    required
/>

<input
    bind:value={formData.patient_profile!.medical_record_number}
    placeholder="Medical Record Number"
    name="date_of_birth"
    class="input"
    required
/>