<script lang="ts">
	import api from '$lib/api';

	const { caseId, onUploaded } = $props();
	let file: File | null = $state(null);
	let loading = $state(false);
	let previewUrl: string | null = $state(null);
	let fileInputRef: HTMLInputElement;

	async function upload() {
		try {
			if (!file) throw new Error('Please select a scan file.');
			loading = true;
			
			const formData = new FormData();
			formData.append('image', file);
			
			const response = await api.uploadFile(`/cases/${caseId}/scans/upload/`, file);
			if (!response.ok) throw new Error('Upload failed.');
			
			// Reset form
			file = null;
			previewUrl = null;
			fileInputRef.value = '';
			
			onUploaded();
		} catch (err) {
			console.log(err);
		} finally {
			loading = false;
		}
	}

	function handleFileSelect(selectedFile: File | null) {
		if (!selectedFile) return;
		
		file = selectedFile;
		
		const reader = new FileReader();
		reader.onload = (e) => previewUrl = e.target?.result as string;
		reader.readAsDataURL(selectedFile);
	}

	function removeFile() {
		file = null;
		previewUrl = null;
		fileInputRef.value = '';
	}
</script>

<div class="bg-card rounded-lg border p-6 space-y-6">
    <div class="flex items-center gap-2">
        <svg class="w-5 h-5 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
        </svg>
        <h3 class="text-lg font-semibold">Upload New Scan</h3>
    </div>

    {#if !file}
        <!-- Upload Zone -->
        <!-- svelte-ignore a11y_click_events_have_key_events -->
		<!-- svelte-ignore a11y_no_static_element_interactions -->
        <div 
            class="border-2 border-dashed border-border rounded-lg p-8 text-center hover:border-primary/50 hover:bg-muted/30 transition-colors cursor-pointer"
            onclick={() => fileInputRef.click()}
        >
            <div class="w-12 h-12 mx-auto mb-4 rounded-full bg-primary/10 flex items-center justify-center">
                <svg class="w-6 h-6 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
            </div>
            <p class="font-medium mb-1">Choose scan file</p>
            <p class="text-sm text-muted-foreground">JPG, PNG or drag & drop</p>
        </div>
    {:else}
        <!-- File Preview -->
        <div class="border border-border rounded-lg p-3">
            <div class="flex items-center justify-between pb-3 border-b-1">
                <div class="flex items-center gap-3">
                    <div class="w-8 h-8 rounded bg-primary/10 flex items-center justify-center">
                        <svg class="w-4 h-4 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                        </svg>
                    </div>
                    <div>
                        <p class="font-medium text-sm">{file.name}</p>
                        <p class="text-xs text-muted-foreground">{(file.size / (1024 * 1024)).toFixed(2)} MB</p>
                    </div>
                </div>
                <!-- svelte-ignore a11y_consider_explicit_label -->
                <button onclick={removeFile} class="text-muted-foreground hover:text-destructive p-1">
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                    </svg>
                </button>
            </div>
            
            {#if previewUrl}
                <img src={previewUrl} alt="Preview" class="w-full max-w-sm mx-auto rounded border bg-transparent max-h-[550px] object-contain"/>
            {/if}
        </div>
    {/if}

    <!-- Hidden image input for select image handling -->
    <input
        bind:this={fileInputRef}
        type="file"
        accept="image/*"
        class="hidden"
        onchange={(e) => handleFileSelect((e.target as HTMLInputElement).files?.[0] || null)}
    />

    <!-- Upload Button -->
    {#if file}
        <div class="flex gap-3">
            <button
                onclick={upload}
                disabled={loading}
                class="flex-1 bg-primary text-primary-foreground rounded px-4 py-2 font-medium hover:bg-primary/90 disabled:opacity-50 transition-colors flex items-center justify-center gap-2"
            >
                {#if loading}
                    <svg class="w-4 h-4 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                    </svg>
                    Uploading...
                {:else}
                    Upload Scan
                {/if}
            </button>
            <button onclick={removeFile} class="px-4 py-2 border border-border rounded hover:bg-muted/50 transition-colors">
                Cancel
            </button>
        </div>
    {/if}
</div>