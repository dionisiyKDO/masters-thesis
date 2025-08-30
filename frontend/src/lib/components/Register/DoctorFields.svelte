<script lang="ts">
	import type { FormData } from "$lib/types";

    interface Props { formData: FormData }
    let { formData = $bindable() }: Props = $props()
</script>


<input
    bind:value={formData.doctor_profile!.license_number}
    placeholder="Medical License Number"
    name="license_number"
    class="input"
    required
/>

<input
    bind:value={formData.doctor_profile!.specialization}
    placeholder="Specialization"
    name="specialization"
    class="input"
    required
/>