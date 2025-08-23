<script lang="ts">
    import { user, logout } from '$lib/auth';

    const links = [
        { name: 'Home', href: '/' },
    ];
</script>

<header class="p-6 pb-4 mb-6 bg-(--surface) border-b border-[--border]">
    <nav class="flex items-center justify-between">
        {#if $user}
            <a href="/" class="text-xl font-bold">{$user?.username}'s Blog</a>
        {:else}
            <a href="/" class="text-xl font-bold">Blog</a>
        {/if}
        
        <div class="flex items-center gap-4">
            
            {#each links as link}
                <a href={link.href}>{link.name}</a>
            {/each}

            <div class="border-l border-[--border]">
                {#if $user}
                    <button onclick={logout} class="min-w-16 ml-4">Logout</button>
                {:else}
                    <a href="/auth" class="min-w-16 inline-block ml-4">Login</a>
                {/if}
            </div>
        </div>
    </nav>
</header>
