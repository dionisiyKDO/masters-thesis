<script lang="ts">
	import api from '$lib/api';

    const { caseId, onUploaded } = $props()

	let file: File | null = null;
	let loading = $state(false);

	async function upload() {
		try {
			if (!file) throw new Error('Please select a scan file.');
			loading = true;
			
			const formData = new FormData();
			formData.append('image', file);
			const response = await api.uploadFile(`/cases/${caseId}/scans/upload/`, file);
			if (!response.ok) throw new Error('Upload failed.');
			const data = await response.json();
			onUploaded(data);
		} catch (err) {
			console.log(err);
			return;
		} finally {
			loading = false;
		}
	}
</script>

<div class="bg-card space-y-4 rounded-lg border p-6">
	<h3 class="text-lg font-semibold">Upload New Scan</h3>

	<input type="file" accept="image/*" onchange={(e: any) => (file = e.target.files?.[0] ?? null)} />

	<button
		class="bg-primary text-primary-foreground rounded px-4 py-2 disabled:opacity-50"
		onclick={upload}
		disabled={loading}
	>
		{loading ? 'Uploading...' : 'Upload Scan'}
	</button>
</div>
